import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)


st.set_page_config(layout="wide")

# Initialize session state for the page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'


# Function to change the page
def change_page(page):
    st.session_state.current_page = page

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = [
    "Home", 
    "Upload Dataset", 
    "Select Target + Features",
    "Choose + Run Model",
    "View Results"
]
for page in pages:
    if st.sidebar.button(page, key=f"nav_{page}"):
        change_page(page)

# Function for page navigation
def click_page(page):
    st.session_state.current_page = page

# Display content based on the selected page (Home)
if st.session_state.current_page == 'Home':
    st.title('Home')
    st.button(
        "Upload Dataset",
        on_click=lambda: click_page("Upload Dataset"),
        key="home_upload_btn"  # Unique key
    )


# Display content based on selected page (Upload Dataset)
elif st.session_state.current_page == 'Upload Dataset':
    st.markdown("<h1 style='font-size: 45px; color: white;'>Upload Dataset</h1>", unsafe_allow_html=True)
    csv_dataset = st.file_uploader("Upload a file(CSV):", type=['csv']) # Upload dataset here

    if csv_dataset:
        st.session_state.df = pd.read_csv(csv_dataset) # Define dataset here
        st.button(
            "Select Target and Features",
            on_click=lambda: click_page("Select Target + Features"),
            key="upload_select_btn"
        )
    else:
        st.warning("Please upload a dataset first.")

# Display content based on selected page (Select Target/Features) 
elif st.session_state.current_page == 'Select Target + Features':
    st.title('Select Target + Features')

    df = st.session_state.get("df")

    # Select target variable here
    if df is None:
        st.warning("Missing dataset. Please go back and complete 'Upload Dataset'.")
    else:
        target = st.selectbox(
            "Select 1 target variable:", 
            ["Choose an option"] + sorted(df.columns), 
            key="target_selector")
        st.session_state.target = target if target != "Choose an option" else None # Target variable here

        # Select positive outcome here
        if not st.session_state.target or st.session_state.target == "Choose an option":
            st.warning("Please choose a target variable first.")
        else:
            target_variable = st.session_state.target
            target_classes = df[target_variable].unique()

            positive_outcome = st.selectbox(
                "Identify the positive outcome:",
                ["Choose an option"] + sorted(target_classes),
                key='pos_outcome_select'
            )
            st.session_state.positive_outcome = (
                None if positive_outcome == "Choose an option" else positive_outcome
            ) # Positive outcome here

            # Select features here
            if st.session_state.positive_outcome is None:
                st.warning("Please choose a positive outcome first.")
            else:
                potential_features = [column for column in df.columns if column != target_variable] 
                features = st.multiselect(
                    "Select 1+ features:",
                    sorted(potential_features),
                    key = 'feat_select'
                )
                st.session_state.features = features
                
                # Move to next step here
                if len(st.session_state.features) == 0:
                    st.warning("Please choose 1+ features first.")
                else:
                    not_ready_for_next_step = (
                        (st.session_state.target is None) or
                        (st.session_state.positive_outcome is None) or
                        (len(st.session_state.features) == 0)
                    )
                    st.button(
                        "Choose + Run Model",
                        on_click=lambda: click_page("Choose + Run Model"),
                        key="select_choose_btn",
                        disabled= not_ready_for_next_step
                    )

    

# Display content based on selected page (Choose/Run Model)
elif st.session_state.current_page == 'Choose + Run Model':
    st.title('Choose + Run Model')

    df = st.session_state.get("df")
    target_variable = st.session_state.get("target")
    positive_outcome = st.session_state.get("positive_outcome")
    features = st.session_state.get("features")

    if target_variable is None:
        st.warning("Missing target variable. Please go back and complete 'Select Target + Features'.")
    elif positive_outcome is None:
        st.warning("Missing positive outcome. Please go back and complete 'Select Target + Features'.")
    elif len(features) == 0 or features is None:
        st.warning("Missing features. Please go back and complete 'Select Target + Features'.")
    else: 
        potential_models = [
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine"
        ]
        models = st.multiselect(
            "Select 1+ classification models:",
            sorted(potential_models)
        )
        st.session_state.models = models

        # Train/test split slider
        test_size = st.slider(
            "Adjust Slider for Test \% in Train Test Split:",
            min_value=1,
            max_value=99,
            value=20,
            key= "train_test_slider"
        ) / 100.0
        st.session_state.test_size = test_size

        # Random state integer input
        random_state = st.number_input(
            "Type a random state (an integer):",
            min_value=0,
            step=1, 
            value=0,
            key="random_state_input"
        )
        st.session_state.random_state = random_state

        # Run models if selected
        if len(st.session_state.models) == 0:
            st.warning("Please select 1+ classification models.")
        elif st.button("Run Model(s)", key="run_models_btn"):
            X = df[features]
            y = (df[target_variable] == positive_outcome).astype(int)

            # Do train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state = random_state
            )
            
            results = {}

            # Function to fit model
            def fit_model(
                X_train, 
                y_train,
                X_test,
                y_test,
                model_name="Logistic Regression"
            ):
                if model_name == "Logistic Regression":
                    model = LogisticRegression()
                elif model_name == "Random Forest":
                    model = RandomForestClassifier()
                elif model_name == "Support Vector Machine":
                    model = SVC(probability=True)
                 
                model.fit(X_train, y_train)

                # Function to run model
                def run_model(X, y, split_name):
                    y_pred = model.predict(X)
                    y_prob = model.predict_proba(X)[:, 1]

                    results = {
                        "y": y,
                        "y_pred": y_pred,
                        "y_prob": y_prob,
                        f"{split_name}_accuracy": accuracy_score(y, y_pred),
                        f"{split_name}_precision": precision_score(y, y_pred),
                        f"{split_name}_recall": recall_score(y, y_pred),
                        f"{split_name}_f1": f1_score(y, y_pred),
                        f"{split_name}_roc_auc": roc_auc_score(y, y_pred),
                        f"{split_name}_report": classification_report(y, y_pred, output_dict=True)
                    }

                    return results

                train_results = run_model(X_train, y_train, "train")
                test_results = run_model(X_test, y_test, "test")

                return {**train_results, **test_results}

        
            # Fit and run models here
            for model_name in st.session_state.models:
                model_results = fit_model(
                    X_train, 
                    y_train,
                    X_test,
                    y_test,
                    model_name=model_name
                )
                results[model_name] = model_results

            st.session_state.results = results

            # If models were run, view results
            if len(st.session_state.results) == 0:
                st.warning("Please run model(s) first.")
            else:
                not_ready_for_view_results = len(st.session_state.results) == 0

                st.button(
                    "View Results",
                    on_click=lambda: click_page("View Results"),
                    key="view_results_btn",
                    disabled= not_ready_for_view_results
                )

# Display content based on selected page (View Results)
elif st.session_state.current_page == 'View Results':
    st.title('View Results')

    results = st.session_state.get("results", {})
    if not results:
        st.warning("Missing results. Please go back and complete 'Choose + Run Model'.")
    else:
        # Plot ROC curve here
        st.subheader("Receiver Operating Characteristic (ROC) Curve(s):")
        plt.style.use('dark_background')
        colors = ["royalblue", "limegreen", "darkorange"]
        color_idx = 0
        fig, ax = plt.subplots()


        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(
                result["y"],
                result["y_prob"]
            )
            test_auc_score = result["test_roc_auc"]
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {test_auc_score:.2f})", color=colors[color_idx])
            color_idx += 1
        
        ax.plot([0, 1], [0, 1], "k--", color='gray')
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title("ROC Curve(s)")
        ax.legend()
        st.pyplot(fig)

        # Plot Metrics table here
        st.subheader("Performance Metrics:")
        train_or_test_select = st.radio(
            "Select between train/test metrics:",
            ["Train", "Test"],
            index=1,
            horizontal=True, 
        )
        train_or_test = train_or_test_select.lower()

        metrics_table = {}

        for model_name, result in results.items():
            metrics_table[model_name] = {
                "Accuracy": round(result.get(f"{train_or_test}_accuracy", 0), 2),
                "Precision": round(result.get(f"{train_or_test}_precision", 0), 2),
                "AUC-ROC": round(result.get(f"{train_or_test}_roc_auc", 0), 2)
            }

        st.dataframe(pd.DataFrame(metrics_table))


    st.button(
        "Home",
        on_click=lambda: click_page("Home"),
        key="view_home_btn"
    )

