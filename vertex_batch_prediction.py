def run_batch_prediction():

    # Define the model resource name (use the model ID from the previously registered model)
    model_resource_name = "projects/1058865513354/locations/us-central1/models/7021187685372919808"
    # Input and output Cloud Storage locations
    input_data_gcs_path = "gs://vertex_custom_learning/iris_batch_prediction.csv"  # Path to the CSV or JSONL input file
    output_data_gcs_path = "gs://vertex_custom_learning/batch_output_predictions/"  # Directory to store the predictions

    # Run the batch prediction job
    batch_prediction_job = aiplatform.BatchPredictionJob.create(
        job_display_name="iris-batch-prediction-job",
        model_name=model_resource_name,
        instances_format="csv",  # or 'jsonl' depending on your input file format
        gcs_source=input_data_gcs_path,
        gcs_destination_prefix=output_data_gcs_path,
        machine_type="n1-standard-4",  # Customize the machine type as per your needs
    )
    
    batch_prediction_job.wait()  # Wait for the job to finish

    print(f"Batch prediction job completed. Predictions saved to {output_data_gcs_path}")
