from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_session import Session
from model_handler import ModelHandler
import os
from config import Config

# Create the Flask application instance
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Load configuration
app.config.from_object(Config)

# Ensure the secret key is set
if not app.secret_key:
    app.secret_key = os.urandom(24)
    print("Warning: Using a randomly generated secret key. Sessions will be lost on restart.")

# Initialize the session
Session(app)

# Initialize the model handler
model_handler = ModelHandler(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get conversation history from session at the beginning of the function
    # This ensures it's always defined regardless of execution path
    conversation_history = session.get('conversation_history', [])
    
    if request.method == 'POST':
        model_family = request.form.get('model_family')
        model_name = request.form.get('model_name')
        
        # Add diagnostic logging
        app.logger.info(f"Model selection: family={model_family}, name={model_name}")
        
        # Validate model selection
        if model_family == 'openai':
            if model_name not in ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo']:
                flash('Invalid OpenAI model selected', 'error')
                return redirect(url_for('index'))
        elif model_family == 'anthropic':
            # Using the corrected Claude model names
            valid_anthropic_models = [
                'claude-3-opus-20240229',
                'claude-3-5-haiku-20241022',
                'claude-3-7-sonnet-20250219'
            ]
            if model_name not in valid_anthropic_models:
                flash(f'Invalid Anthropic model selected: {model_name}', 'error')
                return redirect(url_for('index'))
        else:
            flash('Invalid model family selected', 'error')
            return redirect(url_for('index'))
        
        # Store selected model in session
        session['model_family'] = model_family
        session['model_name'] = model_name
        return redirect(url_for('index'))
    
    # Set default model if not already set
    if 'model_family' not in session:
        session['model_family'] = 'openai'
    if 'model_name' not in session:
        session['model_name'] = 'gpt-4o' if session['model_family'] == 'openai' else 'claude-3-7-sonnet-20250219'
    
    return render_template('index.html', 
                          conversation_history=conversation_history,
                          model_family=session.get('model_family'),
                          model_name=session.get('model_name'),
                          config=app.config)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('user_message')
    if not user_message:
        flash('Please enter a message', 'error')
        return redirect(url_for('index'))
    
    model_family = session.get('model_family', 'openai')
    model_name = session.get('model_name', 'gpt-4o' if model_family == 'openai' else 'claude-3-7-sonnet-20250219')
    
    # Get conversation history
    conversation_history = session.get('conversation_history', [])
    
    try:
        # Process the request through the model handler
        assistant_message = model_handler.process_request(
            model_family=model_family,
            model_name=model_name,
            user_message=user_message,
            conversation_history=conversation_history
        )
        
        # Add to conversation history
        conversation_history.append({
            'user_message': user_message,
            'assistant_message': assistant_message
        })
        
        # Update session
        session['conversation_history'] = conversation_history
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        flash(f"Error: {str(e)}", 'error')
    
    return redirect(url_for('index'))

@app.route('/clear', methods=['POST'])
def clear_conversation():
    session['conversation_history'] = []
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Only warn about missing API keys if neither is available
    if not app.config.get('OPENAI_API_KEY') and not app.config.get('ANTHROPIC_API_KEY'):
        app.logger.error("No API keys configured. At least one provider API key is required.")
    elif not app.config.get('OPENAI_API_KEY'):
        app.logger.warning("OpenAI API key not found. OpenAI models will not be available.")
    elif not app.config.get('ANTHROPIC_API_KEY'):
        app.logger.warning("Anthropic API key not found. Anthropic models will not be available.")
    
    app.run(debug=True)
