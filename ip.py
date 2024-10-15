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

from google.cloud import aiplatform

def initialize_vertex_ai():
    aiplatform.init(
        project="nileshproject-435805",
        location="us-central1",
        staging_bucket="gs://vertex_custom_learning"
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

    print(f"Model training completed. Model resource name: {model.resource_name}")
    return job

def register_model(training_job):
    serving_container_image_uri = "us-central1-docker.pkg.dev/nileshproject-435805/vertex-custom/iris_model@sha256:f0b4a6674840117aff5ca0ebdbcbede51041c5099d71eeb869f2e5decf85d80b"
    
    model_uri = training_job.get_model_artifact_uri()
    
    registered_model = aiplatform.Model.upload(
        display_name="iris-registered-model-1",
        artifact_uri=model_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )
    
    print(f"Model registered with name: {registered_model.display_name}")
    print(f"Model artifact URI: {model_uri}")
    return registered_model

def deploy_model(model):
    endpoint = aiplatform.Endpoint.create(display_name="iris-model-endpoint-1")
    
    deployed_model = endpoint.deploy(
        model=model,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=2,
    )
    
    print(f"Model deployed to endpoint: {endpoint.resource_name}")
    return endpoint

def make_prediction(endpoint):
    instance = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    prediction = endpoint.predict([instance])
    print(f"Prediction result: {prediction}")

def main():
    initialize_vertex_ai()
    
    # Train the model
    training_job = train_model()
    
    # Register the model
    registered_model = register_model(training_job)
    
    # Deploy the model
    endpoint = deploy_model(registered_model)
    
    # Make a prediction (optional)
    make_prediction(endpoint)
    
    # Uncomment these lines if you want to clean up resources
    # endpoint.undeploy_all()
    # endpoint.delete()

if __name__ == "__main__":
    main()