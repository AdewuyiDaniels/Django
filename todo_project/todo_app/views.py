# todo_app/views.py

from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Task
import speech_recognition as sr
from transformers import BertForConditionalGeneration, BertTokenizer

# Load the pre-trained BERT model and tokenizer
model = BertForConditionalGeneration.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def generate_response(voice_input):
    # Tokenize the voice input
    input_ids = tokenizer.encode(voice_input, return_tensors='pt')

    # Generate a response using the BERT model
    response_ids = model.generate(input_ids)

    # Decode the generated response
    generated_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return generated_response

def convert_text_to_speech(text):
    # Implement text-to-speech functionality using a service or library of your choice
    # This example assumes you have a function that converts text to speech
    # Replace it with an appropriate implementation
    speech_response = text  # Placeholder, replace with actual text-to-speech logic

    return speech_response

def index(request):
    tasks = Task.objects.all()
    return render(request, 'todo_app/index.html', {'tasks': tasks})

def add_task(request):
    if request.method == 'POST':
        description = request.POST.get('task')
        if description:
            Task.objects.create(description=description)
            return JsonResponse({'result': 'success'})
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
