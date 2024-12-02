from prometheus_client import Counter, Histogram, Gauge, start_http_server
import tensorflow.lite as tflite
import gradio as gr
import numpy as np
import cv2
import time

# ==========================
# Prometheus Metrics
# ==========================
# Metrics for inference
INFERENCE_COUNTER = Counter('inference_count', 'Number of inferences', ['model', 'input_type'])
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Distribution of inference latencies', ['model', 'input_type'], buckets=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, float("inf")])
INFERENCE_CONFIDENCE = Gauge('inference_confidence', 'Confidence of predictions', ['model', 'predicted_class'])

# Metrics for user feedback
FEEDBACK_COUNTER = Counter('feedback_count', 'Number of feedback entries received', ['feedback_type', 'model'])

start_http_server(8000)  # Prometheus metrics available at http://localhost:8000/metrics

# ==========================
# Load TFLite Models
# ==========================
def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load TFLite Models
movenet_thunder = load_model("./models/movenet_thunder.tflite")
movenet_lightning = load_model("./models/movenet_lightning.tflite")
both_model = load_model("./models/best_both.tflite")
agnostic_model = load_model("./models/best_agnostic.tflite")
cognizant_model = load_model("./models/best_cognizant.tflite")

class_names_both = ["arms_crossed", "down", "head_both", "hips_both", "out_both", "up_both"]
class_names_agnostic = ["arms_crossed", "down", "head_both", "head_one", "hips_both", "hips_one", "out_both", "out_one", "up_both", "up_one"]
class_names_cognizant = ["arms_crossed", "down", "head_both", "head_left", "head_right", "hips_both", "hips_left", "hips_right", "out_both", "out_left", "out_right", "up_both", "up_left", "up_right"]

# ==========================
# Inference Functions
# ==========================
def run_movenet(image, model="Thunder"):
    """
    Runs MoveNet inference using the selected model (Thunder or Lightning).

    Args:
        image (np.ndarray): Preprocessed input image.
        model (str): Model type ("Thunder" or "Lightning").

    Returns:
        np.ndarray: MoveNet output tensor.
    """
    if model == "Thunder":
        interpreter = movenet_thunder
        input_size = 256
    elif model == "Lightning":
        interpreter = movenet_lightning
        input_size = 192
    else:
        raise ValueError("Invalid model type. Choose 'Thunder' or 'Lightning'.")

    # Resize the input image
    resized_image = cv2.resize(image, (input_size, input_size))

    # Prepare the input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(resized_image, axis=0).astype(np.uint8))

    # Run inference
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def classify_pose(interpreter, class_names, keypoints):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = keypoints.flatten().astype(np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_details[0]['index'])[0]
    return {
        "predicted_class": class_names[np.argmax(probabilities)],
        "confidence": np.max(probabilities),
        "probabilities": probabilities.tolist()
    }

def monitored_classification(model_choice, input_data, input_type="image"):
    if model_choice == "Both Model":
        model, class_names = both_model, class_names_both
    elif model_choice == "Hand-Agnostic Model":
        model, class_names = agnostic_model, class_names_agnostic
    elif model_choice == "Hand-Cognizant Model":
        model, class_names = cognizant_model, class_names_cognizant
    else:
        raise ValueError("Invalid model choice")

    INFERENCE_COUNTER.labels(model=model_choice, input_type=input_type).inc()
    start_time = time.time()
    result = classify_pose(model, class_names, input_data)
    latency = time.time() - start_time
    INFERENCE_LATENCY.labels(model=model_choice, input_type=input_type).observe(latency)
    INFERENCE_CONFIDENCE.labels(model=model_choice, predicted_class=result["predicted_class"]).set(result["confidence"])
    return result

# ==========================
# Feedback Collection
# ==========================
class CustomFlagging(gr.FlaggingCallback):
    def __init__(self):
        super().__init__()
        self.flagged_data = []

    def setup(self, dir=None):
        """
        Setup method required by the abstract class.
        It initializes any necessary resources for flagging.
        """
        self.flagged_data = []  # Initialize in-memory storage

    def flag(self, model, feedback_type, expected_output=None, comment=None):
        """
        Record feedback for mismatches or general comments.

        Args:
            model (str): The name of the pose classification model.
            feedback_type (str): The type of feedback ("mismatch" or "comment").
            expected_output (str, optional): The expected pose in case of a mismatch.
            comment (str, optional): A general comment.

        Returns:
            None: Explicitly suppress any return value for Gradio.
        """
        feedback_data = {
            "model": model,
            "feedback_type": feedback_type,
            "expected_output": expected_output,
            "comment": comment,
        }
        FEEDBACK_COUNTER.labels(feedback_type=feedback_type, model=model).inc()
        self.flagged_data.append(feedback_data)

flag_callback = CustomFlagging()

def get_class_names(model_choice):
    """
    Returns the class names based on the selected pose classification model.
    """
    if model_choice == "Both Model":
        return class_names_both
    elif model_choice == "Hand-Agnostic Model":
        return class_names_agnostic
    elif model_choice == "Hand-Cognizant Model":
        return class_names_cognizant
    else:
        return []

# ==========================
# Gradio Interface
# ==========================
def is_person_detected(confidence_scores, threshold):
    """
    Determines if a person is detected based on confidence scores.

    Args:
        confidence_scores (np.ndarray): Confidence scores from MoveNet output tensor.
        threshold (float): Minimum confidence score to consider a person detected.

    Returns:
        bool: True if a person is detected, False otherwise.
    """
    return np.any(confidence_scores > threshold)

def process_image(image, model_choice, threshold, movenet_model):
    """
    Process an uploaded image and return the classification result.
    """
    if image is None:
        return "Please upload an image."  # Handle missing image case

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Run MoveNet inference with the selected model
    output_tensor = run_movenet(image, movenet_model)  # Shape: (1, 1, 17, 3)

    # Check if a person is detected
    if not is_person_detected(output_tensor[0, 0, :, 2], threshold):
        return "Person not found."

    # Flatten the output tensor for pose classification
    input_data = output_tensor.flatten().astype(np.float32)
    result = monitored_classification(model_choice, input_data, input_type="image")
    return f"Predicted Pose: {result['predicted_class']} (Confidence: {result['confidence']:.2f})"

def process_webcam_live(frame, model_choice, threshold, movenet_model):
    """
    Process live frames from the webcam and return the classification result.
    """
    if frame is None:
        return "No frame received."

    # Mirror and preprocess the frame
    frame = cv2.flip(frame, 1)

    # Run MoveNet inference with the selected model
    output_tensor = run_movenet(frame, movenet_model)  # Shape: (1, 1, 17, 3)

    # Check if a person is detected
    if not is_person_detected(output_tensor[0, 0, :, 2], threshold):
        return "Person not found."

    # Flatten the output tensor for pose classification
    input_data = output_tensor.flatten().astype(np.float32)
    result = monitored_classification(model_choice, input_data, input_type="webcam")
    return f"Predicted Pose: {result['predicted_class']} (Confidence: {result['confidence']:.2f})"

iface = gr.Blocks()
with iface:
    with gr.Tab("Image Upload"):
        image_input = gr.Image(sources=['upload'], label="Upload Image")
        model_choice = gr.Radio(["Both Model", "Hand-Agnostic Model", "Hand-Cognizant Model"], value="Hand-Agnostic Model", label="Choose Pose Classification Model")
        movenet_model_choice = gr.Radio(["Thunder", "Lightning"], label="Choose MoveNet Model", value="Thunder")
        threshold_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Confidence Threshold")
        classify_btn = gr.Button("Classify Pose")
        output_text = gr.Textbox(label="Classification Result")

        # Feedback section
        expected_output = gr.Dropdown([], label="Expected Pose (for mismatch flagging)", allow_custom_value=True)
        flag_mismatch_btn = gr.Button("Flag Mismatch")
        general_comment = gr.Textbox(label="General Comments", lines=3)
        flag_comment_btn = gr.Button("Submit Comment")

        # Update the expected output dropdown dynamically
        model_choice.change(
            lambda choice: gr.update(choices=get_class_names(choice)),
            inputs=model_choice,
            outputs=expected_output,
        )

        classify_btn.click(
            process_image,
            inputs=[image_input, model_choice, threshold_slider, movenet_model_choice],
            outputs=output_text,
        )

        flag_mismatch_btn.click(
            fn=lambda model, expected_output: (
                flag_callback.flag(model, "mismatch", expected_output=expected_output),
                gr.update(value=""),  # Clear the expected output dropdown
                gr.update(value=""),  # Clear classification result
            ),
            inputs=[model_choice, expected_output],
            outputs=[expected_output, output_text],
        )

        flag_comment_btn.click(
            fn=lambda model, comment: (
                flag_callback.flag(model, "comment", comment=comment),
                gr.update(value=""),  # Clear the general comment textbox
                gr.update(value=""),  # Clear classification result
            ),
            inputs=[model_choice, general_comment],
            outputs=[general_comment, output_text],
        )

    with gr.Tab("Webcam Stream"):
        webcam_stream = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam")
        model_choice_webcam = gr.Radio(["Both Model", "Hand-Agnostic Model", "Hand-Cognizant Model"], value="Hand-Agnostic Model", label="Choose Pose Classification Model")
        movenet_model_choice_webcam = gr.Radio(["Thunder", "Lightning"], label="Choose MoveNet Model", value="Thunder")
        threshold_slider_webcam = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Confidence Threshold")
        classification_result = gr.Textbox(label="Pose Classification Result", lines=2)

        # Feedback section
        expected_output_webcam = gr.Dropdown([], label="Expected Pose (for mismatch flagging)", allow_custom_value=True)
        flag_mismatch_btn_webcam = gr.Button("Flag Mismatch")
        general_comment_webcam = gr.Textbox(label="General Comments", lines=3)
        flag_comment_btn_webcam = gr.Button("Submit Comment")

        # Update the expected output dropdown dynamically
        model_choice_webcam.change(
            lambda choice: gr.update(choices=get_class_names(choice)),
            inputs=model_choice_webcam,
            outputs=expected_output_webcam,
        )

        webcam_stream.stream(
            fn=process_webcam_live,
            inputs=[webcam_stream, model_choice_webcam, threshold_slider_webcam, movenet_model_choice_webcam],
            outputs=classification_result,
        )

        flag_mismatch_btn_webcam.click(
            fn=lambda model, expected_output: (
                flag_callback.flag(model, "mismatch", expected_output=expected_output),
                gr.update(value=""),  # Clear the expected output dropdown
                gr.update(value=""),  # Clear classification result
            ),
            inputs=[model_choice_webcam, expected_output_webcam],
            outputs=[expected_output_webcam, classification_result],
        )

        flag_comment_btn_webcam.click(
            fn=lambda model, comment: (
                flag_callback.flag(model, "comment", comment=comment),
                gr.update(value=""),  # Clear the general comment textbox
                gr.update(value=""),  # Clear classification result
            ),
            inputs=[model_choice_webcam, general_comment_webcam],
            outputs=[general_comment_webcam, classification_result],
        )

    # Populate the expected output dropdown initially
    iface.load(
        lambda: gr.update(choices=get_class_names("Hand-Agnostic Model")),  # Default model choice
        inputs=None,
        outputs=expected_output,
    )

iface.launch(server_name="0.0.0.0", server_port=7860)
