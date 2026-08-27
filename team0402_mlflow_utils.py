import os
import boto3
import mlflow
from botocore.exceptions import ClientError


MLFLOW_APP_ARN = os.getenv(
    "TEAM04_MLFLOW_APP_ARN",
    "arn:aws:sagemaker:ap-southeast-1:044528205969:mlflow-app/app-ANFQ3RACFV2G"
)


def initialize_mlflow(
    student_id,
    experiment_name,
    team_id="team04",
    project_name="credit-card-fraud-detection",
    region="ap-southeast-1"
):
    """
    Connect to the existing SageMaker MLflow App,
    configure the MLflow experiment, and generate
    a fresh presigned MLflow UI URL.
    """

    print(f"Initializing SageMaker MLflow connection for {student_id}...")
    print(f"Target Experiment: {experiment_name}")

    sm = boto3.client("sagemaker", region_name=region)

    app_arn = MLFLOW_APP_ARN

    print(f"MLflow App ARN: {app_arn}")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(app_arn)

    # Set experiment
    mlflow.set_experiment(experiment_name)

    # Generate fresh MLflow UI URL
    try:
        url_response = sm.create_presigned_mlflow_app_url(
            Arn=app_arn,
            ExpiresInSeconds=300,
            SessionExpirationDurationInSeconds=3600,
        )

        print("MLflow Tracking URI successfully set.")
        print(
            "Fresh MLflow UI URL:\n"
            f"{url_response['AuthorizedUrl']}"
        )

    except ClientError as e:
        print(
            "Could not generate presigned MLflow UI URL. "
            "Tracking may still be configured."
        )
        print(e)

    return app_arn