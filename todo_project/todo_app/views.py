# todo_app/views.py

from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Task
import speech_recognition as sr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from gtts import gTTS
from io import BytesIO
from django.http import JsonResponse

# Load the pre-trained DistilBERT model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
nlu_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

def generate_response(voice_input):
    # Tokenize the voice input
    input_ids = tokenizer.encode(voice_input, return_tensors='pt')

    # Generate a response using DistilBERT (classification)
    output = model(input_ids)

    # For simplicity, let's extract the logit corresponding to the [CLS] token
    # You might need to fine-tune DistilBERT for better text generation results
    logit_cls = output.logits[:, 0]

    # Use the logit for some logic or convert it to text
    generated_response = "Your logic or text generation here."

    return generated_response

def convert_text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    speech_bytes = BytesIO()
    tts.save(speech_bytes)
    
    return speech_bytes.getvalue()

def index(request):
    tasks = Task.objects.all()
    return render(request, 'todo_app/index.html', {'tasks': tasks})

def extract_task_description(user_input):
    try:
        # Tokenize and format user input
        inputs = tokenizer(user_input, return_tensors="pt")

        # Forward pass through the NLU model
        outputs = nlu_model(**inputs)

        # Interpret the output and extract relevant information
        predicted_label = outputs.logits.argmax().item()
        
        # In a real-world scenario, you would map predicted labels to specific task descriptions
        task_description = "Implement logic to map labels to task descriptions"

        return task_description
    except Exception as e:
        # Handle unexpected errors during NLU processing
        print(f"Error in NLU processing: {e}")
        return "Error in NLU processing"

def add_task(request):
    if request.method == 'POST':
        user_input = request.POST.get('task')
        
        if user_input:
            # Extract task description using NLU with error handling
            description = extract_task_description(user_input)

            if "Error" in description:
                # Return an error response if NLU processing fails
                return JsonResponse({'result': 'error', 'message': description})

            try:
                # Check if a task with the same description already exists
                existing_task = Task.objects.get(description=description)
                return JsonResponse({'result': 'error', 'message': 'Task already exists'})
            except ObjectDoesNotExist:
                # Task does not exist, create a new one with timestamp
                new_task = Task.objects.create(description=description)

                # Include additional details like timestamp
                timestamp = new_task.created_at.strftime("%Y-%m-%d %H:%M:%S")

                return JsonResponse({'result': 'success', 'timestamp': timestamp})
        else:
            return JsonResponse({'result': 'error', 'message': 'Task cannot be empty'})
    return redirect('index')

def delete_task(request, task_id):
    try:
        task = Task.objects.get(id=task_id)
        task.delete()
        return JsonResponse({'result': 'success'})
    except Task.DoesNotExist:
        return JsonResponse({'result': 'error', 'message': 'Task not found'})

def voice_input(request):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)

    try:
        task = recognizer.recognize_google(audio)
        
        # Generate a response using BERT
        generated_response = generate_response(task)

        # Convert the text response to speech
        speech_response = convert_text_to_speech(generated_response)

        # Create a task with the recognized text
        Task.objects.create(description=task)

        return JsonResponse({'result': 'success', 'task': task, 'response': generated_response, 'speech_response': speech_response})
    except sr.UnknownValueError:
        return JsonResponse({'result': 'error', 'message': 'Could not understand audio'})
    except sr.RequestError as e:
        return JsonResponse({'result': 'error', 'message': f"Could not request results from Google Speech Recognition service: {e}"})
def switch_mode(request):
    # Implement any additional logic needed when switching between listening and responding modes
    return JsonResponse({'result': 'success'})