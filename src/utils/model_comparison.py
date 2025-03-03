# src/utils/model_comparison.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import json


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
            metrics_df = pd.read_csv(metrics_path)

            # Convert to dictionary for easier access
            metrics_dict = {
                row["Metric"]: row["Value"] for _, row in metrics_df.iterrows()
            }

            # Store data
            self.models_data[model_name] = {
                "metrics": metrics_dict,
                "confusion_matrix_path": confusion_matrix_path,
                "class_names": class_names,
            }

            # Add model_info if provided
            if model_info:
                self.models_data[model_name]["model_info"] = model_info

            print(f"Added results for {model_name}")
        except Exception as e:
            print(f"Error adding results for {model_name}: {str(e)}")

    def add_model_from_evaluation_dir(self, model_name, eval_dir):
        """
        Add model results from evaluation directory

        Args:
            model_name (str): Name of the model
            eval_dir (str): Path to evaluation directory
        """
        metrics_path = os.path.join(eval_dir, "metrics.csv")
        confusion_matrix_path = None

        # Try to find confusion matrix data
        viz_dir = os.path.join(eval_dir, "visualizations")
        if os.path.exists(viz_dir):
            # Check for saved confusion matrix data
            cm_data_path = os.path.join(viz_dir, "confusion_matrix_data.npy")
            if os.path.exists(cm_data_path):
                confusion_matrix_path = cm_data_path

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

        self.add_model_results(
            model_name, metrics_path, confusion_matrix_path, class_names
        )

    def scan_evaluation_directories(self, base_dir="reports/evaluations"):
        """
        Scan and add results from all evaluation directories

        Args:
            base_dir (str): Base directory containing evaluation results
        """
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

            for metric in self.metrics:
                if metric in model_data["metrics"]:
                    row[metric.capitalize()] = model_data["metrics"][metric]

            data.append(row)

        df = pd.DataFrame(data)
        return df

    def plot_metrics_comparison(self):
        """
        Generate bar plots comparing model metrics

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get comparison data
        comparison_df = self.generate_comparison_table()

        # Update this line to use lowercase metric names to match self.metrics
        melted_df = pd.melt(
            comparison_df,
            id_vars=["Model"],
            value_vars=[m.capitalize() for m in self.metrics],
            var_name="Metric",
            value_name="Value",
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create grouped bar plot
        sns.barplot(x="Model", y="Value", hue="Metric", data=melted_df, ax=ax)

        # Customize plot
        plt.title("Model Performance Comparison", fontsize=16)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Metric")
        plt.tight_layout()

        return fig

    def get_best_model(self, metric="f1"):
        """
        Get the best performing model based on specified metric

        Args:
            metric (str): Metric to use for comparison

        Returns:
            tuple: (model_name, metric_value)
        """
        comparison_df = self.generate_comparison_table()
        metric_col = metric.capitalize()

        if metric_col not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in model data")

        # Find row with highest metric value
        best_row = comparison_df.loc[comparison_df[metric_col].idxmax()]

        return best_row["Model"], best_row[metric_col]

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

        # Generate comparison table
        comparison_df = self.generate_comparison_table()
        comparison_df.to_csv(
            os.path.join(report_dir, "metrics_comparison.csv"), index=False
        )

        # Generate comparison plots
        fig = self.plot_metrics_comparison()
        fig.savefig(
            os.path.join(report_dir, "metrics_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Determine best model for each metric
        best_models = {}
        for metric in self.metrics:
            try:
                model_name, value = self.get_best_model(metric)
                best_models[metric] = {"model": model_name, "value": value}
            except Exception as e:
                print(f"Could not determine best model for {metric}: {str(e)}")

        # Generate HTML report
        html_report_path = os.path.join(report_dir, "report.html")

        with open(html_report_path, "w") as f:
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
                    
                    <h2>Performance Metrics Comparison</h2>
                    <img src="metrics_comparison.png" alt="Metrics Comparison">
                    
                    <h2>Metrics Table</h2>
                    <table border="1">
                        <tr>
                            {"".join(f"<th>{col}</th>" for col in comparison_df.columns)}
                        </tr>
                        {"".join(
                            f"<tr>{''.join(f'<td>{cell}</td>' for cell in row.values)}</tr>"
                            for _, row in comparison_df.iterrows()
                        )}
                    </table>
                    
                    <h2>Best Performing Models</h2>
                    <table border="1">
                        <tr>
                            <th>Metric</th>
                            <th>Best Model</th>
                            <th>Value</th>
                        </tr>
                        {"".join(
                            f"<tr><td>{metric.capitalize()}</td><td>{data['model']}</td><td>{data['value']:.4f}</td></tr>"
                            for metric, data in best_models.items()
                        )}
                    </table>
            """
            )

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
                    viz_path = os.path.join(
                        "reports",
                        "evaluations",
                        best_model_name,
                        "visualizations",
                        "confusion_matrix.png",
                    )
                    if os.path.exists(viz_path):
                        # Copy file to report directory
                        import shutil

                        shutil.copy(
                            viz_path,
                            os.path.join(report_dir, "best_model_confusion_matrix.png"),
                        )

                        f.write(
                            """
                        <h3>Confusion Matrix</h3>
                        <img src="best_model_confusion_matrix.png" alt="Confusion Matrix">
                        """
                        )

                    # Add misclassification examples if available
                    misclass_path = os.path.join(
                        "reports",
                        "evaluations",
                        best_model_name,
                        "visualizations",
                        "misclassified_examples.png",
                    )
                    if os.path.exists(misclass_path):
                        # Copy file to report directory
                        import shutil

                        shutil.copy(
                            misclass_path,
                            os.path.join(
                                report_dir, "best_model_misclassifications.png"
                            ),
                        )

                        f.write(
                            """
                        <h3>Common Misclassifications</h3>
                        <img src="best_model_misclassifications.png" alt="Misclassifications">
                        """
                        )

            # Close HTML document
            f.write(
                """
                </div>
            </body>
            </html>
            """
            )

        print(f"Report generated successfully: {html_report_path}")
        return html_report_path
