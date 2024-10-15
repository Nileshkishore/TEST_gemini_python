# from google.cloud import aiplatform

# # Initialize Vertex AI environment and specify the GCS staging bucket
# aiplatform.init(
#     project="nileshproject-435805", 
#     location="us-central1",
#     #staging_bucket="gs://vertex_custom_learning"  # GCS bucket for staging files
# )

# # Define the custom container training job
# job = aiplatform.CustomContainerTrainingJob(
#     display_name="iris-custom-training-job",  # Display name for your training job
#     container_uri="us-central1-docker.pkg.dev/nileshproject-435805/vertex-custom/iris_model:2",  # Your custom container URI for training
#     #model_serving_container_image_uri="us-central1-docker.pkg.dev/nileshproject-435805/vertex-custom/iris_model@sha256:f0b4a6674840117aff5ca0ebdbcbede51041c5099d71eeb869f2e5decf85d80b"  # Container for serving the model
#     staging_bucket="gs://vertex_custom_learning"  # GCS bucket for staging files
# )

# # Run the training job
# model = job.run(
#     replica_count=1,
#     machine_type="n1-standard-4"      # Adjust based on your resource requirements
#     #model_display_name="iris-trained-model",  # Display name for the model to be registered
# )

# from google.cloud import aiplatform

# def initialize_vertex_ai():
#     aiplatform.init(
#         project="nileshproject-435805",
#         location="us-central1",
#         staging_bucket="gs://vertex_custom_learning"
#     )

# def train_model():
#     job = aiplatform.CustomContainerTrainingJob(
#         display_name="iris-custom-training-job-1",
#         container_uri="us-central1-docker.pkg.dev/nileshproject-435805/vertex-custom/iris_model:2",
#         staging_bucket="gs://vertex_custom_learning"
#     )

#     model = job.run(
#         replica_count=1,
#         machine_type="n1-standard-4",
#     )

#     print(f"Model training completed")
#     return model

# def register_model(model):
#     serving_container_image_uri = "us-central1-docker.pkg.dev/nileshproject-435805/vertex-custom/iris_model:2"
    
#     artifact_uri = "gs://vertex_custom_learning"
    
#     registered_model = aiplatform.Model.upload(
#         display_name="iris-registered-model-1",
#         artifact_uri=artifact_uri,
#         serving_container_image_uri=serving_container_image_uri,
#     )
    
#     print("Model registered")
#     return registered_model

# def deploy_model(registered_model):
    
#     endpoint = registered_model.deploy(machine_type="n1-standard-4")
#     return endpoint


# def main():
#     initialize_vertex_ai()
    
#     # Train the model
#     model = train_model()
    
#     # Register the model
#     registered_model = register_model(model)
    
#     # Deploy the model
#     endpoint = deploy_model(registered_model)
#     print(endpoint)

#     # Uncomment these lines if you want to clean up resources
#     # endpoint.undeploy_all()
#     # endpoint.delete()

# if __name__ == "__main__":
#     main()




from google.cloud import aiplatform
from google.auth import load_credentials_from_file

def initialize_vertex_ai():
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
def train_model():
    job = aiplatform.CustomContainerTrainingJob(
        display_name="iris-custom-training-job-1",
        container_uri="us-central1-docker.pkg.dev/nileshproject-435805/vertex-custom/iris_model:2",
        staging_bucket="gs://vertex_custom_learning"
    )

    model = job.run(
        replica_count=1,
        machine_type="n1-standard-4",
    )

    print(f"Model training completed")
    return model

def register_model(model):
    serving_container_image_uri = "us-central1-docker.pkg.dev/nileshproject-435805/vertex-custom/iris_model:2"
    
    artifact_uri = "gs://vertex_custom_learning"
    
    registered_model = aiplatform.Model.upload(
        display_name="iris-registered-model-1",
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
    )
    
    print("Model registered")
    return registered_model

def deploy_model(registered_model):
    endpoint = aiplatform.Endpoint.create(display_name="iris-model-endpoint")
    
    traffic_split = {"0": 100}
    deployed_model = endpoint.deploy(
        model=registered_model,
        deployed_model_display_name="iris-deployed-model-1",
        machine_type="n1-standard-4",
        traffic_split=traffic_split,
        min_replica_count=1,
        max_replica_count=2
    )
    return endpoint

def main():
    initialize_vertex_ai()
    
    # Train the model
    model = train_model()
    
    # Register the model
    registered_model = register_model(model)
    
    # Deploy the model
    endpoint = deploy_model(registered_model)
    print(f"Endpoint: {endpoint.resource_name}")

    # Uncomment these lines if you want to clean up resources
    # endpoint.undeploy_all()
    # endpoint.delete()

if __name__ == "__main__":
    main()