import gemini_functions
import database_functions


# Function to get field value outputs
def generate_final_field_values(question:str):
    
    model_output = gemini_functions.generate_field_values(question=question)
    
    split_token = '<split-token>'
    parts = model_output.split(split_token)
    
    while len(parts) != 2:
        model_output = gemini_functions.generate_field_values(question=question)
        parts = model_output.split(split_token)
    
    image = parts[0].strip().lower() == 'true'
    plot = parts[1].strip().lower() == 'true'
    
    return {
        "Image": image,
        "Plot": plot,
    }
    
    
    
    
# Function to get general output
def generate_final_general_output(question:str):
    model_output = gemini_functions.generate_general_output(question=question)
    data = database_functions.execute_query(model_output)
    return {
        'SQL Query' : model_output,
        'Data' : data
    }
    
    


# Function to get image output
def generate_final_image_output(question:str):
    
    model_output = gemini_functions.generate_image_output(question=question)
    split_token = '<split-token>'
    parts = model_output.split(split_token)
    
    while len(parts) != 2:
        model_output = gemini_functions.generate_image_output(question=question)
        parts = model_output.split(split_token)
    
    query = parts[0].strip()
    statement = parts[1].strip()
    if query.lower() == 'none':
        query = None 

    return {
        'SQL Query' : query,
        'Statement' : statement
    }
    
    

# Function to process plot outputs
def generate_final_plot_output(question:str):
    
    model_output = gemini_functions.generate_plot_output(question=question)
    split_token = '<split-token>'
    parts = model_output.split(split_token) 
    
    while len(parts) != 6:
        model_output = gemini_functions.generate_plot_output(question=question)
        parts = model_output.split(split_token)     
    
    # Process for dict
    plots = ['histogram', 'bar-plot', 'line-plot', 'pie-chart', 'scatter-plot']
    query = parts[0].strip()
    plot_type = parts[1].strip().lower()
    title = parts[2].strip()
    xlabel = parts[3].strip()
    ylabel = parts[4].strip()
    statement = parts[5].strip()
    
    if plot_type == 'pie-chart':
        xlabel = None
        ylabel = None
    
    # Create and return dictionary
    return {
        'SQL Query' : query,
        'Plot type' : plot_type,
        'Title' : title,
        'Xlabel' : xlabel,
        'Ylabel' : ylabel,
        'Statement' : statement
    }
    
    
    
    
    
# Function to generate all the final outputs
def generate_final_module_output(question:str):
    
    field_values = generate_final_field_values(question=question) 
    
    if field_values['Image']:
        img_model = generate_final_image_output(question=question)
        output_img = database_functions.retrieve_images(img_model['SQL Query'])
        img_output = {
            'SQL Query' : img_model['SQL Query'],
            'Statement' : img_model['Statement'],
            'Image' : output_img
        }
    else:
        img_output = None
    
    # For plots
    if field_values['Plot']:
        plot_model = generate_final_plot_output(question=question)
        output_plot = database_functions.execute_plots(plot_model)
        plot_output = {
            'SQL Query' : plot_model['SQL Query'],
            'Statement' : plot_model['Statement'],
            'Plot' : output_plot
        }
    else:
        plot_output = None
    
    # For general outputs
    if not(field_values['Image'] or field_values['Plot'] or field_values['Comparison'] or field_values['Summary']):
        general_output = generate_final_general_output(question=question)
    else:
        general_output = None
    
    
    # Final output
    final_output = {
        'Field Values' : field_values,
        'General' : general_output,
        'Image' : img_output,
        'Plot' : plot_output,
    }
    
    return final_output 
    
    
