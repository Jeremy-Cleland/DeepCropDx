# src/utils/model_comparison.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import json
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelComparisonReport:
    """
    Utility for aggregating and comparing metrics across multiple models
    """

    def __init__(self, output_dir="reports/comparisons", report_id=None):
        self.output_dir = output_dir
        self.models_data = {}
        self.metrics = ["accuracy", "precision", "recall", "f1"]
        self.report_id = report_id
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            f"ModelComparisonReport initialized with output directory: {output_dir}"
        )

    def add_model_results(
        self,
        model_name,
        metrics_path,
        confusion_matrix_path=None,
        class_names=None,
        model_info=None,
    ):
        """
        Add a model's evaluation results to the comparison

        Args:
            model_name (str): Name of the model
            metrics_path (str): Path to CSV file with metrics
            confusion_matrix_path (str, optional): Path to confusion matrix data
            class_names (list, optional): List of class names
            model_info (dict, optional): Additional model information
        """
        # Read metrics from CSV
        try:
            if not os.path.exists(metrics_path):
                logger.error(f"Metrics file not found: {metrics_path}")
                return

            metrics_df = pd.read_csv(metrics_path)
            logger.info(f"Read metrics from {metrics_path} with {len(metrics_df)} rows")

            # Convert to dictionary for easier access
            metrics_dict = {}
            for _, row in metrics_df.iterrows():
                # Ensure metric names are lowercase for consistency
                metric_name = (
                    row["Metric"].lower() if "Metric" in row else row["metric"].lower()
                )
                metrics_dict[metric_name] = (
                    row["Value"] if "Value" in row else row["value"]
                )

            # Store data
            self.models_data[model_name] = {
                "metrics": metrics_dict,
                "confusion_matrix_path": confusion_matrix_path,
                "class_names": class_names,
            }

            # Add model_info if provided
            if model_info:
                self.models_data[model_name]["model_info"] = model_info

            logger.info(f"Added results for {model_name} with metrics: {metrics_dict}")
        except Exception as e:
            logger.error(f"Error adding results for {model_name}: {str(e)}")

    def add_model_from_evaluation_dir(self, model_name, eval_dir):
        """
        Add model results from evaluation directory

        Args:
            model_name (str): Name of the model
            eval_dir (str): Path to evaluation directory
        """
        logger.info(f"Adding model from evaluation directory: {eval_dir}")
        metrics_path = os.path.join(eval_dir, "metrics.csv")
        confusion_matrix_path = None

        if not os.path.exists(metrics_path):
            logger.error(f"Metrics file not found: {metrics_path}")
            return

        # Try to find confusion matrix data
        viz_dir = os.path.join(eval_dir, "visualizations")
        if os.path.exists(viz_dir):
            # Check for saved confusion matrix data
            cm_data_path = os.path.join(viz_dir, "confusion_matrix_data.npy")
            if os.path.exists(cm_data_path):
                confusion_matrix_path = cm_data_path

            # Also look for confusion matrix image
            cm_img_path = os.path.join(viz_dir, "confusion_matrix.png")
            if os.path.exists(cm_img_path):
                confusion_matrix_path = cm_img_path

        # Try to find class names
        class_names = None
        class_mapping_path = os.path.join(
            os.path.dirname(eval_dir), "class_mapping.txt"
        )
        if os.path.exists(class_mapping_path):
            class_names = {}
            with open(class_mapping_path, "r") as f:
                for line in f:
                    if line.strip():
                        name, idx = line.strip().split(",")
                        class_names[int(idx)] = name
            # Convert to list
            if class_names:
                class_names = [class_names[i] for i in range(len(class_names))]

        # If class_mapping.txt is not found, try to find class names from json file
        if not class_names:
            summary_path = os.path.join(eval_dir, "evaluation_summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r") as f:
                        summary_data = json.load(f)
                        if "class_names" in summary_data:
                            class_names = summary_data["class_names"]
                except Exception as e:
                    logger.error(f"Error reading evaluation summary: {str(e)}")

        model_info = {"evaluation_directory": eval_dir}

        self.add_model_results(
            model_name, metrics_path, confusion_matrix_path, class_names, model_info
        )

    def scan_evaluation_directories(self, base_dir="reports/evaluations"):
        """
        Scan and add results from all evaluation directories

        Args:
            base_dir (str): Base directory containing evaluation results
        """
        logger.info(f"Scanning evaluation directories in: {base_dir}")
        if not os.path.exists(base_dir):
            logger.error(f"Base directory not found: {base_dir}")
            return

        for model_dir in os.listdir(base_dir):
            full_path = os.path.join(base_dir, model_dir)
            if os.path.isdir(full_path):
                self.add_model_from_evaluation_dir(model_dir, full_path)

    def generate_comparison_table(self):
        """
        Generate a comparison table of model metrics

        Returns:
            pandas.DataFrame: Comparison table
        """
        data = []

        for model_name, model_data in self.models_data.items():
            row = {"Model": model_name}

            # Add each metric with consistent keys
            for metric in self.metrics:
                if metric in model_data["metrics"]:
                    row[metric.capitalize()] = model_data["metrics"][metric]

            data.append(row)

        df = pd.DataFrame(data)
        logger.info(
            f"Generated comparison table with {len(df)} models and columns: {df.columns.tolist()}"
        )
        return df

    def plot_metrics_comparison(self):
        """
        Generate bar plots comparing model metrics

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get comparison data
        comparison_df = self.generate_comparison_table()

        if comparison_df.empty:
            logger.error("Cannot generate plot: comparison data is empty")
            return None

        # Make sure all expected columns exist
        for metric in [m.capitalize() for m in self.metrics]:
            if metric not in comparison_df.columns:
                logger.warning(f"Metric column '{metric}' not found in comparison data")
                comparison_df[metric] = np.nan

        # Melt the dataframe for plotting
        try:
            melted_df = pd.melt(
                comparison_df,
                id_vars=["Model"],
                value_vars=[m.capitalize() for m in self.metrics],
                var_name="Metric",
                value_name="Value",
            )
        except Exception as e:
            logger.error(f"Error melting dataframe: {str(e)}")
            return None

        # Create figure
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x="Model", y="Value", hue="Metric", data=melted_df, ax=ax)
            plt.title("Model Performance Comparison", fontsize=16)
            plt.ylabel("Score", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Metric")
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            return None

    def get_best_model(self, metric="f1"):
        """
        Get the best performing model based on specified metric

        Args:
            metric (str): Metric to use for comparison

        Returns:
            tuple: (model_name, metric_value)
        """
        comparison_df = self.generate_comparison_table()
        if comparison_df.empty:
            logger.error("Cannot get best model: comparison data is empty")
            return None, None

        metric_col = metric.capitalize()

        if metric_col not in comparison_df.columns:
            logger.error(f"Metric '{metric_col}' not found in comparison data")
            return None, None

        # Find row with highest metric value
        try:
            best_idx = comparison_df[metric_col].idxmax()
            best_row = comparison_df.loc[best_idx]
            logger.info(
                f"Best model for {metric}: {best_row['Model']} with value {best_row[metric_col]}"
            )
            return best_row["Model"], best_row[metric_col]
        except Exception as e:
            logger.error(f"Error finding best model for {metric}: {str(e)}")
            return None, None

    def generate_report(
        self, title="Model Comparison Report", include_best_model_details=True
    ):
        """
        Generate a comprehensive comparison report

        Args:
            title (str): Report title
            include_best_model_details (bool): Whether to include detailed info about best model

        Returns:
            str: Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = self.report_id if self.report_id else timestamp
        report_dir = os.path.join(self.output_dir, f"comparison_{report_id}")
        os.makedirs(report_dir, exist_ok=True)
        logger.info(f"Generating report in directory: {report_dir}")

        # Generate comparison table
        comparison_df = self.generate_comparison_table()

        if comparison_df.empty:
            logger.error("Cannot generate report: comparison data is empty")
            return None

        csv_path = os.path.join(report_dir, "metrics_comparison.csv")
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics comparison to: {csv_path}")

        # Generate comparison plots
        fig = self.plot_metrics_comparison()
        if fig:
            plot_path = os.path.join(report_dir, "metrics_comparison.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved metrics plot to: {plot_path}")

        # Determine best model for each metric
        best_models = {}
        for metric in self.metrics:
            try:
                model_name, value = self.get_best_model(metric)
                if model_name and value is not None:
                    best_models[metric] = {"model": model_name, "value": value}
            except Exception as e:
                logger.error(f"Could not determine best model for {metric}: {str(e)}")

        # Generate HTML report
        html_report_path = os.path.join(report_dir, "report.html")

        try:
            with open(html_report_path, "w") as f:
                # Write HTML header
                f.write(
                    f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>{title}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333366; }}
                        h2 {{ color: #333366; margin-top: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                        th, td {{ text-align: left; padding: 12px; }}
                        th {{ background-color: #333366; color: white; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                        .metric-highlight {{ font-weight: bold; color: #006600; }}
                        .content {{ max-width: 1200px; margin: 0 auto; }}
                        .timestamp {{ color: #666; font-size: 0.8em; }}
                        img {{ max-width: 100%; height: auto; margin-top: 20px; }}
                    </style>
                </head>
                <body>
                    <div class="content">
                        <h1>{title}</h1>
                        <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                """
                )

                # Add metrics comparison plot
                if fig:
                    f.write(
                        f"""
                        <h2>Performance Metrics Comparison</h2>
                        <img src="metrics_comparison.png" alt="Metrics Comparison">
                    """
                    )

                # Add metrics table using pandas HTML
                f.write(
                    f"""
                        <h2>Metrics Table</h2>
                        {comparison_df.to_html(index=False)}
                """
                )

                # Add best models section
                if best_models:
                    f.write(
                        f"""
                        <h2>Best Performing Models</h2>
                        <table border="1">
                            <tr>
                                <th>Metric</th>
                                <th>Best Model</th>
                                <th>Value</th>
                            </tr>
                    """
                    )

                    for metric, data in best_models.items():
                        f.write(
                            f"""
                            <tr>
                                <td>{metric.capitalize()}</td>
                                <td>{data['model']}</td>
                                <td>{data['value']:.4f}</td>
                            </tr>
                        """
                        )

                    f.write("</table>")

                # Add best model details if requested
                if include_best_model_details and best_models.get("f1"):
                    best_model_name = best_models["f1"]["model"]
                    best_model_data = self.models_data.get(best_model_name)

                    if best_model_data:
                        f.write(
                            f"""
                            <h2>Best Overall Model: {best_model_name}</h2>
                            <p>Based on F1 Score: {best_models["f1"]["value"]:.4f}</p>
                        """
                        )

                        # Add confusion matrix visualization if available
                        eval_dir = best_model_data.get("model_info", {}).get(
                            "evaluation_directory", ""
                        )
                        if eval_dir:
                            # Check for confusion matrix in visualizations directory
                            viz_dir = os.path.join(eval_dir, "visualizations")
                            if os.path.exists(viz_dir):
                                # Try multiple possible filenames
                                cm_files = [
                                    "confusion_matrix.png",
                                    "normalized_confusion_matrix.png",
                                    "cm.png",
                                ]
                                for cm_file in cm_files:
                                    viz_path = os.path.join(viz_dir, cm_file)
                                    if os.path.exists(viz_path):
                                        # Copy file to report directory
                                        target_path = os.path.join(
                                            report_dir,
                                            "best_model_confusion_matrix.png",
                                        )
                                        shutil.copy(viz_path, target_path)

                                        f.write(
                                            f"""
                                            <h3>Confusion Matrix</h3>
                                            <img src="best_model_confusion_matrix.png" alt="Confusion Matrix">
                                        """
                                        )
                                        break

                            # Check for misclassification examples
                            for misclass_file in [
                                "misclassified_examples.png",
                                "misclassifications.png",
                            ]:
                                misclass_path = os.path.join(viz_dir, misclass_file)
                                if os.path.exists(misclass_path):
                                    # Copy file to report directory
                                    target_path = os.path.join(
                                        report_dir, "best_model_misclassifications.png"
                                    )
                                    shutil.copy(misclass_path, target_path)

                                    f.write(
                                        f"""
                                        <h3>Common Misclassifications</h3>
                                        <img src="best_model_misclassifications.png" alt="Misclassifications">
                                    """
                                    )
                                    break

                # Close HTML document
                f.write(
                    """
                    </div>
                </body>
                </html>
                """
                )

            logger.info(f"Report generated successfully: {html_report_path}")
            return html_report_path

        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return None


# Helper function to create a comparison report from the command line
def create_comparison_report(
    evaluations_dir="reports/evaluations",
    output_dir="reports/comparisons",
    report_title="Model Performance Comparison",
):
    """Create and generate a model comparison report from existing evaluation directories"""
    report = ModelComparisonReport(output_dir=output_dir)
    report.scan_evaluation_directories(base_dir=evaluations_dir)
    return report.generate_report(title=report_title)


if __name__ == "__main__":
    # This allows running the module directly to generate a comparison report
    report_path = create_comparison_report()
    if report_path:
        print(f"Report generated successfully at: {report_path}")
    else:
        print("Failed to generate comparison report")
