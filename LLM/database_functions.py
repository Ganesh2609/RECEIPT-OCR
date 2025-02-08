import mysql.connector 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
import os
import io


def execute_query(sql_query):

    load_dotenv()
    host = 'localhost'
    user = 'root'
    password = os.getenv('ROOT_PASSWORD')
    database = 'BillDBMS'
    
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        auth_plugin='mysql_native_password'
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    df = pd.DataFrame(rows)
    return df



# Function to execute image query and return output
def retrieve_images(query:str):
    img_blob = execute_query(query)['IMAGE'].iloc[0]
    try:
        img_data = io.BytesIO(img_blob)
        img = Image.open(img_data)
        return img
    except Exception as e:
        print("Error displaying image:", e)
        return None


def fig2img(fig): 
    buf = io.BytesIO() 
    fig.savefig(buf, format='png')
    buf.seek(0) 
    img = Image.open(buf) 
    plt.close(fig)
    return img 


def execute_plots(model_output:dict):
    
    query = model_output['SQL Query']
    data = execute_query(query)
    
    plot_type = model_output['Plot type']
    statement = model_output['Statement']
    
    fig = plt.figure(figsize=(10,6))
    
    try:
        if plot_type == 'histogram':
            plt.bar(data[list(data.columns)[0]], data[list(data.columns)[1]])
        elif plot_type == 'bar-plot':
            plt.bar(data[list(data.columns)[0]], data[list(data.columns)[1]])
        elif plot_type == 'line-plot':
            plt.plot(data[list(data.columns)[0]], data[list(data.columns)[1]])
        elif plot_type == 'pie-chart':
            plt.pie(data[list(data.columns)[1]], labels=data[list(data.columns)[0]], autopct='%1.1f%%')
        elif plot_type == 'scatter-plot':
            plt.scatter(data[list(data.columns)[0]], data[list(data.columns)[1]])
            
        plt.title(model_output['Title'])
        if model_output['Xlabel'] is not None:
            plt.xlabel(model_output['Xlabel'])
            plt.ylabel(model_output['Ylabel'])
            
        return fig2img(fig)

    except Exception as e:
        plt.close(fig)
        print("Error creating plot:", e)
        return None