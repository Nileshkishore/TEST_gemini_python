# from google.cloud import aiplatform

# # Initialize Vertex AI with your project and region
# PROJECT_ID = "nileshproject-435805"
# REGION = "us-central1"  # Adjust based on the region of your endpoint
# ENDPOINT_ID = "2541456256418185216"

# aiplatform.init(project=PROJECT_ID, location=REGION)

# # Load the endpoint by ID
# endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}")

# # Sample input for your model (adapt according to your model's input format)
# #instances = [{"sepal_length": "5.1", "sepal_width": "3.5", "petal_length":"1.4", "petal_width":"0.2"}]
# instances = [[5.1, 3.5, 1.4, 0.2]] 
# # Send the prediction request to the endpoint
# response = endpoint.predict(instances=instances)

# # Print the response from the model
# print(f"Predictions: {response.predictions}")


from google.cloud import aiplatform
from google.api_core.exceptions import FailedPrecondition

# Initialize Vertex AI with your project and region
PROJECT_ID = "nileshproject-435805"
REGION = "us-central1"  # Adjust based on the region of your endpoint
ENDPOINT_ID = "7041678184068153344"

# Initialize the Vertex AI platform
aiplatform.init(project=PROJECT_ID, location=REGION)

try:
    # Load the endpoint by ID (no need for network argument)
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
    )

    # Sample input for your model (adapt according to your model's input format)
    instances = [[5.1, 3.5, 1.4, 0.2]]  # Example input for an Iris model (adjust if needed)

    # Send the prediction request to the endpoint
    response = endpoint.predict(instances=instances)

    # Print the response from the model
    print(f"Predictions: {response.predictions}")

except FailedPrecondition as e:
    print(f"Error occurred: {e}")
    print("This may be due to incorrect endpoint configuration or network issues.")
except Exception as ex:
    print(f"An unexpected error occurred: {ex}")
