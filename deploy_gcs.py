from google.cloud import aiplatform
from google.auth import load_credentials_from_file
 # Path to your service account key JSON file
service_account_file = "/home/sigmoid/Documents/TEST_gemini_python/service-account-key.json"

# Load credentials from your service account key file
credentials, _ = load_credentials_from_file(service_account_file)

# Initialize Vertex AI with the custom service account credentials
aiplatform.init(
        project="nileshproject-435805",
        location="us-central1",
        staging_bucket="gs://vertex_custom_learning",
        credentials=credentials  # Use the custom credentials
)

def deploy_model():
    model = aiplatform.Model.upload(
        display_name="iris-model",
        artifact_uri="gs://vertex_custom_learning/",  # Path to the model in GCS
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
    )

    # Deploy the model to an endpoint
    endpoint = model.deploy(
        machine_type="n1-standard-4"
    )

    print(f"Model deployed to endpoint {endpoint.resource_name}")


deploy_model()