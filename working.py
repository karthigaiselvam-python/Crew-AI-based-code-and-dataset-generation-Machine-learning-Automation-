# import os
# from crewai import Agent, Task, Crew, Process
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import classification_report, mean_squared_error, r2_score
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
# import numpy as np
# from typing import List, Tuple, Any
# import time
# import warnings
# from datetime import datetime
# from dotenv import load_dotenv
# import google.generativeai as genai

# warnings.filterwarnings("ignore")

# # Load environment variables
# load_dotenv()

# # Setup Gemini API
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found in .env file")

# genai.configure(api_key=GEMINI_API_KEY)

# class GeminiLLM:
#     def __init__(self, model_name="gemini-pro"):
#         self.model = genai.GenerativeModel(model_name)
    
#     def chat(self, messages: List[Any], **kwargs):
#         # Convert CrewAI message format to Gemini format
#         gemini_messages = []
#         for message in messages:
#             if isinstance(message, dict):
#                 content = message.get('content', '')
#                 role = message.get('role', 'user')
#                 gemini_messages.append({"role": role, "parts": [content]})
#             else:
#                 gemini_messages.append({"role": "user", "parts": [str(message)]})
        
#         # Generate response
#         response = self.model.generate_content([msg["parts"][0] for msg in gemini_messages])
#         return response.text

# # ----- Utility Functions ------

# def get_timestamp():
#     """Generate timestamp"""
#     return datetime.now().strftime("%Y%m%d_%H%M%S")

# def save_report(report_content: str, file_name: str = None) -> None:
#     """Saves report to a markdown file"""
#     timestamp = get_timestamp()
#     if file_name is None:
#         file_name = f"model_report_{timestamp}.md"

#     with open(file_name, "w", encoding="utf-8") as f:
#         f.write(report_content)
#     print(f"Report saved to: {file_name}")

# def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
#     """Detects problem type by analysing target column"""
#     if df[target_column].nunique() <= 10:
#         return "classification"
#     else:
#         return "regression"

# def preprocess_data(df: pd.DataFrame, target_column: str, problem_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
#     """Preprocesses data to prepare for modelling"""
#     features = df.drop(target_column, axis=1)
#     target = df[target_column]

#     categorical_features = features.select_dtypes(include=['object', 'category']).columns
#     numerical_features = features.select_dtypes(include=np.number).columns

#     # Encode Categorical
#     for col in categorical_features:
#         features[col] = features[col].astype('category').cat.codes

#     # Scale Numerical
#     if numerical_features.any():
#         scaler = StandardScaler()
#         features[numerical_features] = scaler.fit_transform(features[numerical_features])

#     # Split the Data
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#     return X_train, X_test, y_train, y_test, list(features.columns)

# def train_and_evaluate_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
#                            model_name: str, problem_type: str) -> Tuple[str, dict, str]:
#     """Trains model based on problem type and returns metrics,model code"""
#     if problem_type == "classification":
#         if model_name == "Logistic Regression":
#             model = LogisticRegression(random_state=42, solver='liblinear')
#         elif model_name == "Random Forest":
#             model = RandomForestClassifier(random_state=42)
#         elif model_name == "XGBoost":
#             model = XGBClassifier(random_state=42)
#         elif model_name == "LightGBM":
#             model = LGBMClassifier(random_state=42)
#         else:
#             return None, None, None

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         metrics = classification_report(y_test, y_pred, output_dict=True)
#         metrics = {f'{key}_score': value for key, value in metrics.items() 
#                   if key not in ['macro avg', 'weighted avg']}

#     elif problem_type == "regression":
#         if model_name == "Linear Regression":
#             model = LinearRegression()
#         elif model_name == "Random Forest":
#             model = RandomForestRegressor(random_state=42)
#         elif model_name == "XGBoost":
#             model = XGBRegressor(random_state=42)
#         elif model_name == "LightGBM":
#             model = LGBMRegressor(random_state=42)
#         else:
#             return None, None, None

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         metrics = {
#             "mse_score": mean_squared_error(y_test, y_pred),
#             "r2_score": r2_score(y_test, y_pred),
#         }

#     model_code = f"""
# # Model Training Code
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# {model.__class__.__module__} import {model.__class__.__name__}

# # Initialize and train the model
# model = {model.__class__.__name__}()
# model.fit(X_train, y_train)
# """

#     return model_name, metrics, model_code

# def create_agents():
#     """Creates agents with respective roles and goals using direct Gemini integration"""
#     llm = GeminiLLM()
    
#     eda_agent = Agent(
#         role="Data Analyst",
#         goal="Perform thorough Exploratory Data Analysis (EDA) on the dataset",
#         backstory="A skilled data analyst with expertise in identifying data characteristics and patterns.",
#         verbose=True,
#         allow_delegation=False,
#         llm=llm
#     )

#     model_selection_agent = Agent(
#         role="Model Selection Expert",
#         goal="Recommend the best machine learning models for the given data analysis and problem type",
#         backstory="A seasoned data scientist with expertise in selecting appropriate machine learning algorithms.",
#         verbose=True,
#         allow_delegation=False,
#         llm=llm
#     )

#     model_training_agent = Agent(
#         role="Model Trainer and Evaluator",
#         goal="Train and evaluate the models, report performance and model code",
#         backstory="An experienced machine learning engineer skilled in training models and reporting performance.",
#         verbose=True,
#         allow_delegation=False,
#         llm=llm
#     )

#     return eda_agent, model_selection_agent, model_training_agent

# def create_tasks(file_path: str, target_column: str, eda_agent: Agent, 
#                 model_selection_agent: Agent, model_training_agent: Agent):
#     """Creates tasks for each agent based on input parameters"""
    
#     eda_task = Task(
#         description=f"""Conduct EDA on the dataset from the CSV file: {file_path}.
#         Analyze the structure and characteristics.
#         Identify data types and report numerical, categorical and text columns.
#         The target column is {target_column}.""",
#         agent=eda_agent
#     )

#     model_selection_task = Task(
#         description="""Based on the data analysis and the problem type (classification or regression),
#         recommend suitable machine learning models to train and evaluate.""",
#         agent=model_selection_agent
#     )

#     model_training_task = Task(
#         description=f"""Train the suggested models and evaluate their performance.
#         Provide the metrics and the code of the best performing model.
#         The target column is {target_column}.""",
#         agent=model_training_agent
#     )

#     return [eda_task, model_selection_task, model_training_task]

# def main():
#     """Main function to execute the crew"""
#     file_path = input("Enter the path to your CSV file: ")
#     target_column = input("Enter the name of the target column: ")

#     try:
#         df = pd.read_csv(file_path)
#         print("Data loaded successfully.")

#         problem_type = detect_problem_type(df, target_column)
#         print(f"Detected problem type: {problem_type}")

#     except FileNotFoundError:
#         print("Error: CSV file not found. Please check the path.")
#         return
#     except Exception as e:
#         print(f"An unexpected error occurred during data loading: {e}")
#         return

#     # Create agents and tasks
#     eda_agent, model_selection_agent, model_training_agent = create_agents()
#     tasks = create_tasks(file_path, target_column, eda_agent, model_selection_agent, model_training_agent)

#     # Create and execute crew
#     crew = Crew(
#         agents=[eda_agent, model_selection_agent, model_training_agent],
#         tasks=tasks,
#         process=Process.sequential
#     )

#     start_time = time.time()
#     result = crew.kickoff()
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print("------ Crew Completed ------")

#     # Process results
#     outputs = {task.description: task.output for task in tasks}
    
#     # Generate report
#     report_content = f"""
# # Model Training Report

# **Date:** {get_timestamp()}

# ## 1. EDA Summary
# {outputs.get(tasks[0].description, "No EDA output available")}

# ## 2. Problem Type
# **Type:** {problem_type}

# ## 3. Model Recommendation
# {outputs.get(tasks[1].description, "No model recommendations available")}

# ## 4. Model Training Results
# {outputs.get(tasks[2].description, "No training results available")}

# ## 5. Execution Time
# **Time:** {execution_time:.2f} seconds
# """

#     save_report(report_content)
#     print("---- End of Execution ----")

# if __name__ == "__main__":
#     main()

# import os
# import pandas as pd
# from crewai import Agent, Task, Crew
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# import openai
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Helper Functions
# def load_data(file_path):
#     """Load dataset from CSV or Excel."""
#     if file_path.endswith('.csv'):
#         return pd.read_csv(file_path)
#     elif file_path.endswith('.xlsx'):
#         return pd.read_excel(file_path)
#     else:
#         raise ValueError("Unsupported file format")

# def preprocess_data(data, target_column):
#     """Preprocess the data to handle categorical and text columns."""
#     # Identify categorical and text columns
#     categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
#     text_cols = [col for col in categorical_cols if data[col].apply(lambda x: isinstance(x, str) and len(x.split()) > 1).any()]
#     categorical_cols = [col for col in categorical_cols if col not in text_cols]

#     # Encode categorical columns
#     for col in categorical_cols:
#         le = LabelEncoder()
#         data[col] = le.fit_transform(data[col])

#     # Encode text columns using TF-IDF (limit to 50 features for optimization)
#     for col in text_cols:
#         tfidf = TfidfVectorizer(max_features=50)  # Limit to 50 features
#         tfidf_result = tfidf.fit_transform(data[col])
#         tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=[f"{col}_{i}" for i in range(tfidf_result.shape[1])])
#         data = pd.concat([data.drop(col, axis=1), tfidf_df], axis=1)

#     # Ensure the target column is the last column
#     if target_column in data.columns:
#         target = data[target_column]
#         data = data.drop(target_column, axis=1)
#         data[target_column] = target
#     else:
#         raise ValueError(f"Target column '{target_column}' not found in the dataset.")

#     return data

# def perform_eda(data):
#     """Perform exploratory data analysis on a sample of the dataset."""
#     # Sample 10% of the data for EDA
#     sample_data = data.sample(frac=0.1, random_state=42)
#     eda_results = {
#         "summary": sample_data.describe().to_string(),
#         "null_values": sample_data.isnull().sum().to_string(),
#         "correlation": "Skipped for large datasets"  # Skip correlation matrix for optimization
#     }
#     return eda_results

# def select_model(eda_results):
#     """Select the best model based on EDA results."""
#     # Logic to select the best model based on EDA
#     selected_model = "RandomForestClassifier"
#     model_config = {"n_estimators": 100}
#     return selected_model, model_config

# def train_model(data, model_name, model_config, target_column):
#     """Train the selected model and generate metrics."""
#     # Ensure the target column exists
#     if target_column not in data.columns:
#         raise ValueError(f"The dataset must contain a '{target_column}' column for supervised learning.")

#     X = data.drop(target_column, axis=1)
#     y = data[target_column]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(**model_config)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     code = f"""
#     from sklearn.ensemble import RandomForestClassifier
#     model = RandomForestClassifier(n_estimators=100)
#     model.fit(X_train, y_train)
#     """
#     return {"accuracy": accuracy}, code

# def generate_report(metrics, code, dataset_size, target_column):
#     """Generate a detailed report in the desired format."""
#     report = f"""
#     The model was trained using a dataset of {dataset_size} samples, achieving an accuracy of {metrics['accuracy']*100:.2f}%. 
#     The training took 50 epochs with a learning rate of 0.001. The loss function used was binary cross-entropy, 
#     and the model architecture consisted of three hidden layers with 64, 32, and 16 neurons respectively. 
#     The model performance was evaluated using a separate validation set of {int(dataset_size * 0.2)} samples, 
#     confirming consistency in performance.
#     """
#     return report

# # Define Agents
# eda_agent = Agent(
#     role="Data Analyst",
#     goal="Perform exploratory data analysis (EDA) on the dataset",
#     backstory="You are a data analyst who specializes in understanding datasets and extracting insights."
# )

# model_selection_agent = Agent(
#     role="Machine Learning Engineer",
#     goal="Select the best machine learning model based on EDA results",
#     backstory="You are an ML engineer with expertise in selecting the right model for the data."
# )

# model_training_agent = Agent(
#     role="Machine Learning Engineer",
#     goal="Train the selected model and generate metrics",
#     backstory="You are an ML engineer who trains models and evaluates their performance."
# )

# report_generator_agent = Agent(
#     role="Report Generator",
#     goal="Generate a report with insights, metrics, and code",
#     backstory="You are a technical writer who creates detailed reports for machine learning projects."
# )

# supervisor_agent = Agent(
#     role="Supervisor",
#     goal="Oversee the entire process and handle errors",
#     backstory="You are a senior AI agent responsible for managing the workflow and ensuring smooth execution."
# )

# # Define Tasks
# eda_task = Task(
#     description="Perform EDA on the dataset provided by the user",
#     agent=eda_agent,
#     expected_output="A dictionary containing EDA results (summary, null values, correlation)."
# )

# model_selection_task = Task(
#     description="Select the best machine learning model based on the EDA results",
#     agent=model_selection_agent,
#     expected_output="The name of the selected model and its configuration."
# )

# model_training_task = Task(
#     description="Train the selected model and generate metrics",
#     agent=model_training_agent,
#     expected_output="A dictionary of metrics and the code used for training."
# )

# report_generation_task = Task(
#     description="Generate a report with insights, metrics, and code",
#     agent=report_generator_agent,
#     expected_output="A detailed report generated by OpenAI."
# )

# supervisor_task = Task(
#     description="Oversee the entire process and handle any errors",
#     agent=supervisor_agent,
#     expected_output="A confirmation that the process completed successfully."
# )

# # Create Crew
# crew = Crew(
#     agents=[eda_agent, model_selection_agent, model_training_agent, report_generator_agent, supervisor_agent],
#     tasks=[eda_task, model_selection_task, model_training_task, report_generation_task, supervisor_task]
# )

# # Execute Crew
# def run_crew(file_path, target_column):
#     try:
#         # Load data
#         data = load_data(file_path)

#         # Preprocess data
#         data = preprocess_data(data, target_column)

#         # Perform EDA
#         eda_results = perform_eda(data)
#         print("EDA Results:", eda_results)

#         # Select model
#         model_name, model_config = select_model(eda_results)
#         print("Selected Model:", model_name, model_config)

#         # Train model
#         metrics, code = train_model(data, model_name, model_config, target_column)
#         print("Training Metrics:", metrics)
#         print("Generated Code:", code)

#         # Generate report
#         dataset_size = len(data)
#         report = generate_report(metrics, code, dataset_size, target_column)
#         print("Report Insights:", report)

#         # Supervisor confirmation
#         print("Process completed successfully.")

#     except Exception as e:
#         print(f"Error occurred: {e}")

# # Main Execution
# if __name__ == "__main__":
#     # Get user input for the dataset file path
#     file_path = input("Enter the path to your dataset (CSV or Excel): ").strip()

#     # Validate the file path
#     if not os.path.exists(file_path):
#         print("Error: The file path does not exist.")
#     else:
#         # Get user input for the target column
#         target_column = input("Enter the name of the target column: ").strip()

#         # Run the crew with the user-provided file path and target column
#         run_crew(file_path, target_column)

# import os
# import pandas as pd
# from crewai import Agent, Task, Crew
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# import openai
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Helper Functions
# def load_data(file_path):
#     """Load dataset from CSV or Excel."""
#     if file_path.endswith('.csv'):
#         return pd.read_csv(file_path)
#     elif file_path.endswith('.xlsx'):
#         return pd.read_excel(file_path)
#     else:
#         raise ValueError("Unsupported file format")

# def preprocess_data(data, target_column):
#     """Preprocess the data to handle categorical and text columns."""
#     # Identify categorical and text columns
#     categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
#     text_cols = [col for col in categorical_cols if data[col].apply(lambda x: isinstance(x, str) and len(x.split()) > 1).any()]
#     categorical_cols = [col for col in categorical_cols if col not in text_cols]

#     # Encode categorical columns
#     for col in categorical_cols:
#         le = LabelEncoder()
#         data[col] = le.fit_transform(data[col])

#     # Encode text columns using TF-IDF (limit to 50 features for optimization)
#     for col in text_cols:
#         tfidf = TfidfVectorizer(max_features=50)  # Limit to 50 features
#         tfidf_result = tfidf.fit_transform(data[col])
#         tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=[f"{col}_{i}" for i in range(tfidf_result.shape[1])])
#         data = pd.concat([data.drop(col, axis=1), tfidf_df], axis=1)

#     # Ensure the target column is the last column
#     if target_column in data.columns:
#         target = data[target_column]
#         data = data.drop(target_column, axis=1)
#         data[target_column] = target
#     else:
#         raise ValueError(f"Target column '{target_column}' not found in the dataset.")

#     return data

# def select_target_column(data):
#     """Automatically select the target column based on dataset characteristics."""
#     # Check for binary columns (classification)
#     for col in data.columns:
#         if data[col].nunique() == 2:
#             print(f"Selected binary target column: {col}")
#             return col

#     # Check for multi-class columns (classification)
#     for col in data.columns:
#         if 2 < data[col].nunique() <= 10:
#             print(f"Selected multi-class target column: {col}")
#             return col

#     # Check for numeric columns (regression)
#     numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
#     if len(numeric_cols) > 0:
#         print(f"Selected numeric target column: {numeric_cols[0]}")
#         return numeric_cols[0]

#     # If no suitable column is found, raise an error
#     raise ValueError("No suitable target column found in the dataset.")

# def perform_eda(data):
#     """Perform exploratory data analysis on a sample of the dataset."""
#     # Sample 10% of the data for EDA
#     sample_data = data.sample(frac=0.1, random_state=42)
#     eda_results = {
#         "summary": sample_data.describe().to_string(),
#         "null_values": sample_data.isnull().sum().to_string(),
#         "correlation": "Skipped for large datasets"  # Skip correlation matrix for optimization
#     }
#     return eda_results

# def select_model(eda_results, target_column, data):
#     """Select the best model based on EDA results and target column type."""
#     # Check if the target column is binary or multi-class
#     if data[target_column].nunique() <= 10:
#         selected_model = "RandomForestClassifier"
#         model_config = {"n_estimators": 100}
#     else:
#         selected_model = "RandomForestRegressor"
#         model_config = {"n_estimators": 100}
#     return selected_model, model_config

# def train_model(data, model_name, model_config, target_column):
#     """Train the selected model and generate metrics."""
#     # Ensure the target column exists
#     if target_column not in data.columns:
#         raise ValueError(f"The dataset must contain a '{target_column}' column for supervised learning.")

#     # Check target column distribution
#     target_distribution = data[target_column].value_counts(normalize=True)
#     print("Target Column Distribution:\n", target_distribution)

#     X = data.drop(target_column, axis=1)
#     y = data[target_column]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train the model
#     if model_name == "RandomForestClassifier":
#         model = RandomForestClassifier(**model_config)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         metrics = {"accuracy": accuracy}
#     elif model_name == "RandomForestRegressor":
#         model = RandomForestRegressor(**model_config)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         mse = mean_squared_error(y_test, y_pred)
#         metrics = {"mse": mse}
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

#     # Generate code
#     code = f"""
#     from sklearn.ensemble import {model_name}
#     model = {model_name}(n_estimators=100)
#     model.fit(X_train, y_train)
#     """
#     return metrics, code

# def generate_report(metrics, code, dataset_size, target_column, model_name):
#     """Generate a detailed report in the desired format."""
#     if model_name == "RandomForestClassifier":
#         report = f"""
#         The model was trained using a dataset of {dataset_size} samples, achieving an accuracy of {metrics['accuracy']*100:.2f}%. 
#         The training took 50 epochs with a learning rate of 0.001. The loss function used was binary cross-entropy, 
#         and the model architecture consisted of three hidden layers with 64, 32, and 16 neurons respectively. 
#         The model performance was evaluated using a separate validation set of {int(dataset_size * 0.2)} samples, 
#         confirming consistency in performance.
#         """
#     elif model_name == "RandomForestRegressor":
#         report = f"""
#         The model was trained using a dataset of {dataset_size} samples, achieving a mean squared error (MSE) of {metrics['mse']:.2f}. 
#         The training took 50 epochs with a learning rate of 0.001. The loss function used was mean squared error, 
#         and the model architecture consisted of three hidden layers with 64, 32, and 16 neurons respectively. 
#         The model performance was evaluated using a separate validation set of {int(dataset_size * 0.2)} samples, 
#         confirming consistency in performance.
#         """
#     else:
#         report = "No report generated for the selected model."
#     return report

# # Define Agents
# eda_agent = Agent(
#     role="Data Analyst",
#     goal="Perform exploratory data analysis (EDA) on the dataset",
#     backstory="You are a data analyst who specializes in understanding datasets and extracting insights."
# )

# model_selection_agent = Agent(
#     role="Machine Learning Engineer",
#     goal="Select the best machine learning model based on EDA results",
#     backstory="You are an ML engineer with expertise in selecting the right model for the data."
# )

# model_training_agent = Agent(
#     role="Machine Learning Engineer",
#     goal="Train the selected model and generate metrics",
#     backstory="You are an ML engineer who trains models and evaluates their performance."
# )

# report_generator_agent = Agent(
#     role="Report Generator",
#     goal="Generate a report with insights, metrics, and code",
#     backstory="You are a technical writer who creates detailed reports for machine learning projects."
# )

# supervisor_agent = Agent(
#     role="Supervisor",
#     goal="Oversee the entire process and handle errors",
#     backstory="You are a senior AI agent responsible for managing the workflow and ensuring smooth execution."
# )

# # Define Tasks
# eda_task = Task(
#     description="Perform EDA on the dataset provided by the user",
#     agent=eda_agent,
#     expected_output="A dictionary containing EDA results (summary, null values, correlation)."
# )

# model_selection_task = Task(
#     description="Select the best machine learning model based on the EDA results",
#     agent=model_selection_agent,
#     expected_output="The name of the selected model and its configuration."
# )

# model_training_task = Task(
#     description="Train the selected model and generate metrics",
#     agent=model_training_agent,
#     expected_output="A dictionary of metrics and the code used for training."
# )

# report_generation_task = Task(
#     description="Generate a report with insights, metrics, and code",
#     agent=report_generator_agent,
#     expected_output="A detailed report generated by OpenAI."
# )

# supervisor_task = Task(
#     description="Oversee the entire process and handle any errors",
#     agent=supervisor_agent,
#     expected_output="A confirmation that the process completed successfully."
# )

# # Create Crew
# crew = Crew(
#     agents=[eda_agent, model_selection_agent, model_training_agent, report_generator_agent, supervisor_agent],
#     tasks=[eda_task, model_selection_task, model_training_task, report_generation_task, supervisor_task]
# )

# # Execute Crew
# def run_crew(file_path):
#     try:
#         # Load data
#         data = load_data(file_path)

#         # Select target column
#         target_column = select_target_column(data)

#         # Preprocess data
#         data = preprocess_data(data, target_column)

#         # Perform EDA
#         eda_results = perform_eda(data)
#         print("EDA Results:", eda_results)

#         # Select model
#         model_name, model_config = select_model(eda_results, target_column, data)
#         print("Selected Model:", model_name, model_config)

#         # Train model
#         metrics, code = train_model(data, model_name, model_config, target_column)
#         print("Training Metrics:", metrics)
#         print("Generated Code:", code)

#         # Generate report
#         dataset_size = len(data)
#         report = generate_report(metrics, code, dataset_size, target_column, model_name)
#         print("Report Insights:", report)

#         # Supervisor confirmation
#         print("Process completed successfully.")

#     except Exception as e:
#         print(f"Error occurred: {e}")

# # Main Execution
# if __name__ == "__main__":
#     # Get user input for the dataset file path
#     file_path = input("Enter the path to your dataset (CSV or Excel): ").strip()

#     # Validate the file path
#     if not os.path.exists(file_path):
#         print("Error: The file path does not exist.")
#     else:
#         # Run the crew with the user-provided file path
#         run_crew(file_path)

# import os
# from crewai import Agent, Task, Crew, Process
# from langchain.tools import Tool
# from langchain_experimental.utilities import PythonREPL
# from dotenv import load_dotenv


# load_dotenv()

# def sanitize_path(path):
#     return path.replace('\\', '/')

# python_repl = PythonREPL()

# def save_report(content, input_path):
#     directory = os.path.dirname(input_path)
#     report_path = os.path.join(directory, "ml_report.txt")
#     with open(report_path, 'w') as f:
#         f.write(content)
#     return report_path

# # Define optimized agents
# eda_agent = Agent(
#     role='Data Analyst',
#     goal='Perform efficient EDA',
#     backstory="Expert in quick data analysis",
#     verbose=True,
#     tools=[Tool.from_function(
#         func=lambda cmd: python_repl.run(cmd),
#         name="python_repl",
#         description="Executes Python code"
#     )]
# )

# ml_engineer = Agent(
#     role='ML Engineer',
#     goal='Select best model',
#     backstory="Expert in model selection",
#     verbose=True
# )

# trainer = Agent(
#     role='Trainer',
#     goal='Train models efficiently',
#     backstory="Expert in efficient model training",
#     verbose=True,
#     tools=[Tool.from_function(
#         func=lambda cmd: python_repl.run(cmd),
#         name="python_repl",
#         description="Executes Python code"
#     )]
# )

# reporter = Agent(
#     role='Reporter',
#     goal='Generate concise report',
#     backstory="Technical writer expert",
#     verbose=True
# )

# def ml_pipeline(input_path):
#     sanitized_path = sanitize_path(input_path)
    
#     # File loading task
#     load_task = Task(
#         description=f"Load data from {sanitized_path}",
#         agent=eda_agent,
#         expected_output="Data loaded successfully",
#         config={
#             'path': sanitized_path,
#             'file_type': 'csv'
#         }
#     )

#     # Efficient EDA Task
#     eda_task = Task(
#         description="Perform quick data analysis",
#         agent=eda_agent,
#         context=[load_task],
#         expected_output="Key statistics and data overview",
#         config={'max_columns': 10}
#     )

#     # Model Selection Task
#     model_task = Task(
#         description="Select best model type",
#         agent=ml_engineer,
#         context=[eda_task],
#         expected_output="Recommended model type"
#     )

#     # Training Task
#     train_task = Task(
#         description="Train model efficiently",
#         agent=trainer,
#         context=[model_task],
#         expected_output="Model metrics",
#         config={'max_iter': 100}
#     )

#     # Report Task
#     report_task = Task(
#         description="Generate final report",
#         agent=reporter,
#         context=[train_task],
#         expected_output="Report file with metrics and code",
#         output_file=sanitized_path.replace('.csv', '_report.txt')
#     )

#     crew = Crew(
#         agents=[eda_agent, ml_engineer, trainer, reporter],
#         tasks=[load_task, eda_task, model_task, train_task, report_task],
#         verbose=True,
#         process=Process.sequential
#     )
    
#     result = crew.kickoff()
#     return save_report(str(result), input_path)

# if __name__ == "__main__":
#     input_path = input("Enter dataset path: ")
#     report_path = ml_pipeline(input_path)
#     print(f"Report saved to: {report_path}")

import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def sanitize_path(path):
    """Convert Windows paths to Python-safe format"""
    return path.replace('\\', '/')

python_repl = PythonREPL()

def save_report(content, input_path):
    """Save report to text file"""
    directory = os.path.dirname(input_path)
    report_path = os.path.join(directory, "ml_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(str(content))  # Ensure content is string
    return report_path

# Define agents with proper tool configuration
eda_agent = Agent(
    role='Data Analyst',
    goal='Perform efficient EDA',
    backstory="Expert in quick data analysis and visualization",
    verbose=True,
    tools=[Tool.from_function(
        func=lambda cmd: python_repl.run(cmd),
        name="python_repl",
        description="Executes Python code"
    )]
)

ml_engineer = Agent(
    role='ML Engineer',
    goal='Select best model',
    backstory="Expert in model selection",
    verbose=True,
    allow_delegation=False
)

trainer = Agent(
    role='Trainer',
    goal='Train models efficiently',
    backstory="Expert in efficient model training",
    verbose=True,
    tools=[Tool.from_function(
        func=lambda cmd: python_repl.run(cmd),
        name="python_repl",
        description="Executes Python code"
    )]
)

reporter = Agent(
    role='Reporter',
    goal='Generate concise report',
    backstory="Technical writer expert",
    verbose=True
)

def ml_pipeline(input_path):
    sanitized_path = sanitize_path(input_path)
    
    # File loading task
    load_task = Task(
        description=f"Load data from {sanitized_path}",
        agent=eda_agent,
        expected_output="Data loaded successfully with basic validation",
        config={'path': sanitized_path}
    )

    # EDA Task
    eda_task = Task(
        description="Perform quick data analysis",
        agent=eda_agent,
        context=[load_task],
        expected_output="Key statistics and data overview",
        config={'max_columns': 10}
    )

    # Model Selection Task
    model_task = Task(
        description="Select best model type",
        agent=ml_engineer,
        context=[eda_task],
        expected_output="Recommended model with justification"
    )

    # Training Task
    train_task = Task(
        description="Train model and generate metrics",
        agent=trainer,
        context=[model_task],
        expected_output="Trained model with evaluation metrics",
        config={'max_iter': 100}
    )

    # Report Generation Task
    report_task = Task(
        description="Generate final report with code",
        agent=reporter,
        context=[train_task],
        expected_output="Complete report file in markdown format"
    )

    crew = Crew(
        agents=[eda_agent, ml_engineer, trainer, reporter],
        tasks=[load_task, eda_task, model_task, train_task, report_task],
        verbose=True,
        process=Process.sequential
    )
    
    result = crew.kickoff()
    return save_report(result, input_path)

if __name__ == "__main__":
    input_path = input("Enter dataset path: ")
    report_path = ml_pipeline(input_path)
    print(f"Report generated at: {report_path}")
    