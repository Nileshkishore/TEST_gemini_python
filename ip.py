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
