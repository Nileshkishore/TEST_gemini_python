import kfp
from kfp import dsl
from kfp.dsl import component
from google.cloud import aiplatform
from google.auth import load_credentials_from_file

# Define components as functions for each step in the pipeline

@component(base_image="python:3.9")
def train_model_component(
    project_id: str,
    location: str,
    staging_bucket: str,
    service_account_file: str,
    container_uri: str,
    machine_type: str
) -> str:
    """Train a model using Vertex AI CustomContainerTrainingJob."""
    credentials, _ = load_credentials_from_file(service_account_file)
    
    # Initialize Vertex AI environment
    aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket, credentials=credentials)

    # Define and run the training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name="iris-custom-training-job",
        container_uri=container_uri,
        staging_bucket=staging_bucket
    )
    
    model = job.run(replica_count=1, machine_type=machine_type)
    print("Model training completed.")
    
    # Return the resource name of the trained model
    return model.resource_name

@component(base_image="python:3.9")
def register_model_component(
    project_id: str,
    location: str,
    artifact_uri: str,
    serving_container_image_uri: str,
    service_account_file: str
) -> str:
    """Register the trained model in Vertex AI."""
    credentials, _ = load_credentials_from_file(service_account_file)
    
    # Initialize Vertex AI environment
    aiplatform.init(project=project_id, location=location, credentials=credentials)
    
    # Register the model
    registered_model = aiplatform.Model.upload(
        display_name="iris-registered-model-1",
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri
    )
    
    print("Model registered.")
    
    # Return the registered model name
    return registered_model.resource_name

@component(base_image="python:3.9")
def deploy_model_component(
    project_id: str,
    location: str,
    registered_model_name: str,
    service_account_file: str,
    machine_type: str
) -> str:
    """Deploy the registered model to an endpoint in Vertex AI."""
    credentials, _ = load_credentials_from_file(service_account_file)
    
    # Initialize Vertex AI environment
    aiplatform.init(project=project_id, location=location, credentials=credentials)
    
    # Deploy the registered model
    model = aiplatform.Model(model_name=registered_model_name)
    endpoint = model.deploy(machine_type=machine_type, min_replica_count=1, max_replica_count=2)
    
    print(f"Model deployed to endpoint: {endpoint.resource_name}")
    
    # Return the endpoint name
    return endpoint.resource_name


# Define the pipeline that ties all components together
@dsl.pipeline(
    name="vertex-ai-training-deployment-pipeline",
    description="Pipeline to train, register, and deploy a model on Vertex AI using Kubeflow."
)
def vertex_ai_pipeline(
    project_id: str,
    location: str,
    staging_bucket: str,
    service_account_file: str,
    container_uri: str,
    serving_container_image_uri: str,
    artifact_uri: str,
    machine_type: str = "n1-standard-4"
):
    # Step 1: Train the model
    train_model_task = train_model_component(
        project_id=project_id,
        location=location,
        staging_bucket=staging_bucket,
        service_account_file=service_account_file,
        container_uri=container_uri,
        machine_type=machine_type
    )

    # Step 2: Register the trained model
    register_model_task = register_model_component(
        project_id=project_id,
        location=location,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        service_account_file=service_account_file
    ).after(train_model_task)

    # Step 3: Deploy the registered model
    deploy_model_task = deploy_model_component(
        project_id=project_id,
        location=location,
        registered_model_name=register_model_task.output,
        service_account_file=service_account_file,
        machine_type=machine_type
    ).after(register_model_task)


# Compile the pipeline
if __name__ == "__main__":
    # Compile the pipeline into a YAML file for Kubeflow
    kfp.compiler.Compiler().compile(
        pipeline_func=vertex_ai_pipeline,
        package_path="vertex_training_deployment_pipeline.yaml"
    )
