import streamlit as st
import pandas as pd
import json
import os
from streamlit_js_eval import streamlit_js_eval
import sys
sys.path.append('.')
from config.config import get_raw_data_path, get_processed_data_path, get_final_data_path, get_data_path, get_viz_path

# Initialize session state variables
if 'validator_name' not in st.session_state:
    st.session_state.validator_name = None
if 'current_plant_code' not in st.session_state:
    st.session_state.current_plant_code = None
if 'auto_advance' not in st.session_state:
    st.session_state.auto_advance = False
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

def load_special_cases():
    """Load empty and no-content plant codes"""
    try:
        with open('empty_plant_codes.json', 'r') as f:
            empty_codes = json.load(f)
        with open('pc_no_relevant_content.json', 'r') as f:
            no_content_codes = json.load(f)
        return empty_codes, no_content_codes
    except Exception as e:
        st.error(f"Error loading special cases: {str(e)}")
        return [], []

def load_data():
    """Load and return the sample data"""
    try:
        df = pd.read_pickle("data/final/validation_sample_data.pkl")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def load_json_files(plant_code, no_content=False):
    """Load JSON files for a given plant code"""
    try:
        if no_content:
            # Only load article relevance for no-content plants
            with open(f'"data/processed/results/article_relevance/"{plant_code}.json', 'r') as f:
                article_relevance = json.load(f)
            return None, article_relevance, None
        else:
            # Check if all required files exist
            required_files = [
                f'"data/processed/results/search/"{plant_code}.json',
                f'"data/processed/results/article_relevance/"{plant_code}.json',
                f'"data/processed/results/scores/"{plant_code}.json'
            ]
            if not all(os.path.exists(f) for f in required_files):
                st.warning(f"Skipping plant {plant_code} - missing required files")
                return None, None, None
                
            # Load all files for normal plants
            with open(f'"data/processed/results/search/"{plant_code}.json', 'r') as f:
                search_results = json.load(f)
            with open(f'"data/processed/results/article_relevance/"{plant_code}.json', 'r') as f:
                article_relevance = json.load(f)
            with open(f'"data/processed/results/scores/"{plant_code}.json', 'r') as f:
                scores = json.load(f)
            return search_results, article_relevance, scores
    except Exception as e:
        st.warning(f"Skipping plant {plant_code} - {str(e)}")
        return None, None, None

def load_existing_validations():
    """Load existing validation results from the single JSON file"""
    validation_file = 'results/validation/validation_scores.json'
    if os.path.exists(validation_file):
        try:
            with open(validation_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading existing validations: {str(e)}")
            return []
    return []

def save_validation(plant_code, validation_data):
    """Save validation results to the single JSON file"""
    try:
        os.makedirs('results/validation', exist_ok=True)
        validation_file = 'results/validation/validation_scores.json'
        
        # Load existing validations
        existing_validations = load_existing_validations()
        
        # Check if this plant code already exists
        existing_plant_codes = [item['plant_code'] for item in existing_validations]
        if plant_code in existing_plant_codes:
            st.error(f"Validation already exists for plant {plant_code}")
            return False
        
        # Add new validation data
        existing_validations.append(validation_data)
        
        # Save updated data
        with open(validation_file, 'w') as f:
            json.dump(existing_validations, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving validation: {str(e)}")
        return False

def display_validation_instructions():
    """Display validation instructions and definitions"""
    st.sidebar.markdown("## Validation Instructions")
    st.sidebar.markdown("""
    **Validation Categories:**
    
    - **TP (True Positive):** The AI correctly identified something as present/true
    - **FP (False Positive):** The AI incorrectly identified something as present/true when it wasn't
    - **TN (True Negative):** The AI correctly identified something as absent/false  
    - **FN (False Negative):** The AI incorrectly identified something as absent/false when it was present
    
    **For Article Relevance:**
    - **TP:** Article is relevant and AI gave grade 3-5
    - **FP:** Article is irrelevant but AI gave grade 3-5
    - **TN:** Article is irrelevant and AI gave grade 1-2
    - **FN:** Article is relevant but AI gave grade 1-2
    
    **For Project Scores:**
    - **TP:** Score of 1 is correct based on evidence
    - **FP:** Score is 1 when it should be 0 (false positive)
    - **TN:** Score of 0 is correct (no evidence found)
    - **FN:** Score is 0 when it should be 1 (false negative)

    **Reasoning:** Provide a brief explanation (max 1 sentence)
    """)

def display_article_validation(search_results, article_relevance):
    """Display search results with relevance scores and validation inputs"""
    st.subheader("Article Validation")
    
    article_validations = {}
    
    # Create a mapping of article letters to relevance scores
    relevance_scores = {
        score['article_letter']: score 
        for score in article_relevance.get('scores_and_justifications', [])
    }
    
    for idx, result in enumerate(search_results.get('organic', [])):
        article_letter = chr(65 + idx)
        score = relevance_scores.get(article_letter, {})
        
        with st.expander(f"Article {article_letter}: {result.get('title', 'No title')}"):
            st.write(f"URL: [{result.get('link', 'No link')}]({result.get('link', '#')})")
            st.write(f"Relevance Score: {score.get('grade', 'N/A')}")
            st.write(f"Justification: {score.get('justification', 'N/A')}")
            
            # Validation inputs
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                result_val = st.selectbox(
                    "Result", 
                    ["TP", "FP", "TN", "FN"],
                    key=f"article_{article_letter}_result"
                )
            
            with col2:
                confidence = st.slider(
                    "Confidence", 
                    0.0, 1.0, 0.8, 0.1,
                    key=f"article_{article_letter}_confidence"
                )
            
            with col3:
                reasoning = st.text_input(
                    "Reasoning (max 1 sentence)",
                    key=f"article_{article_letter}_reasoning"
                )
            
            article_validations[f"article_{article_letter}"] = {
                "result": result_val,
                "confidence": confidence,
                "reasoning": reasoning
            }
    
    return article_validations

def display_scores_validation(scores):
    """Display project scores with validation inputs"""
    st.subheader("Project Scores Validation")
    
    score_validations = {}
    
    # Define all score variables
    score_variables = [
        'mention_support', 'mention_opp', 'physical_opp', 'policy_opp', 
        'legal_opp', 'opinion_opp', 'environmental_opp', 'participation_opp',
        'tribal_opp', 'health_opp', 'intergov_opp', 'property_opp',
        'compensation', 'delay', 'co_land_use'
    ]
    
    for score_obj in scores.get('all_scores_and_sources', []):
        st.write("---")
        for field, value in score_obj.items():
            if field != 'narrative':
                if isinstance(value, list):
                    # Handle list type scores (like mention_support)
                    for item in value:
                        st.write(f"{field}:")
                        st.write(f"Score: {item.get('score', 'N/A')}")
                        st.write(f"Sources: {item.get('sources', 'N/A')}")
                else:
                    # Handle simple integer scores
                    st.write(f"{field}: {value}")
                
                # Validation inputs for each field
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    result_val = st.selectbox(
                        "Result", 
                        ["TP", "FP", "TN", "FN"],
                        key=f"score_{field}_result"
                    )
                
                with col2:
                    confidence = st.slider(
                        "Confidence", 
                        0.0, 1.0, 0.8, 0.1,
                        key=f"score_{field}_confidence"
                    )
                
                with col3:
                    reasoning = st.text_input(
                        "Reasoning (max 1 sentence)",
                        key=f"score_{field}_reasoning"
                    )
                
                score_validations[field] = {
                    "result": result_val,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
        
        # Display narrative separately
        st.write("---")
        st.write("Narrative:")
        narrative_text = score_obj.get('narrative', 'No narrative available')
        st.write(narrative_text)
        
        # Validation inputs for narrative
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            narrative_result = st.selectbox(
                "Result", 
                ["TP", "FP", "TN", "FN"],
                key="narrative_result"
            )
        
        with col2:
            narrative_confidence = st.slider(
                "Confidence", 
                0.0, 1.0, 0.8, 0.1,
                key="narrative_confidence"
            )
        
        with col3:
            narrative_reasoning = st.text_input(
                "Reasoning (max 1 sentence)",
                key="narrative_reasoning"
            )
        
        score_validations['narrative'] = {
            "result": narrative_result,
            "confidence": narrative_confidence,
            "reasoning": narrative_reasoning
        }
    
    return score_validations

def clear_form_state():
    """Clear all form-related session state variables"""
    for key in list(st.session_state.keys()):
        if key.startswith(('article_', 'score_', 'narrative_')):
            del st.session_state[key]
    st.session_state.form_submitted = False

def create_validation_output(plant_code, article_validations, score_validations):
    """Create validation output in the same format as "data/final/validation_human_labels.json""""
    
    # Create accuracy summary (just the results, no confidence/reasoning)
    accuracy_summary = {}
    
    # Add article validations to accuracy summary (A through I only)
    for article_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        key = f'accuracy_article_{article_letter}'
        article_key = f'article_{article_letter}'
        if article_key in article_validations:
            accuracy_summary[key] = article_validations[article_key]['result']
        else:
            accuracy_summary[key] = ''
    
    # Add score validations to accuracy summary
    score_fields = [
        'mention_support', 'mention_opp', 'physical_opp', 'policy_opp', 
        'legal_opp', 'opinion_opp', 'environmental_opp', 'participation_opp',
        'tribal_opp', 'health_opp', 'intergov_opp', 'property_opp',
        'compensation', 'delay', 'co_land_use'
    ]
    
    for field in score_fields:
        key = f'accuracy_{field}'
        if field in score_validations:
            accuracy_summary[key] = score_validations[field]['result']
        else:
            accuracy_summary[key] = ''
    
    # Add narrative validation to accuracy summary
    if 'narrative' in score_validations:
        accuracy_summary['accuracy_narrative'] = score_validations['narrative']['result']
    else:
        accuracy_summary['accuracy_narrative'] = ''
    
    # Create detailed validation structure
    detailed_validation = {
        'article_validations': article_validations,
        'score_validations': {k: v for k, v in score_validations.items() if k != 'narrative'},
        'narrative_validation': score_validations.get('narrative', {})
    }
    
    # Create final output structure (without raw_response)
    validation_output = {
        'plant_code': plant_code,
        'accuracy_summary': accuracy_summary,
        'detailed_validation': detailed_validation
    }
    
    return validation_output

def main():
    st.title("Renewable Energy Project Validation (Test)")
    
    # Display validation instructions
    display_validation_instructions()
    
    # User identification and settings
    if not st.session_state.validator_name:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.session_state.validator_name = st.text_input("Please enter your name:")
        with col2:
            st.session_state.auto_advance = st.checkbox("Auto-advance to next project", value=False)
        if not st.session_state.validator_name:
            st.warning("Please enter your name to continue")
            return
    
    # Load data and special cases
    df = load_data()
    empty_codes, no_content_codes = load_special_cases()
    if df is None:
        return
    
    # Load existing validations to check what's already been done
    existing_validations = load_existing_validations()
    existing_plant_codes = set(item['plant_code'] for item in existing_validations)
    
    # Filter out empty plants and already validated plants
    unvalidated_df = df[
        (~df['plant_code'].isin(empty_codes)) & 
        (~df['plant_code'].astype(str).isin(existing_plant_codes))
    ]
    
    if unvalidated_df.empty:
        st.success("All entries have been validated!")
        st.write(f"Total validations completed: {len(existing_validations)}")
        return
    
    # Get first unvalidated row that has all required files
    for _, row in unvalidated_df.iterrows():
        plant_code = row['plant_code']
        is_no_content = plant_code in no_content_codes
        
        # Try loading files
        search_results, article_relevance, scores = load_json_files(plant_code, is_no_content)
        
        # If files loaded successfully or this is a no-content plant with article_relevance
        if (not is_no_content and all((search_results, article_relevance, scores))) or \
           (is_no_content and article_relevance is not None):
            current_row = row
            break
    else:
        st.error("No plants with complete files found to validate")
        return
    
    # Update current plant code in session state
    st.session_state.current_plant_code = plant_code
    
    # Display progress
    st.sidebar.markdown(f"**Progress:** {len(existing_validations)} completed, {len(unvalidated_df)} remaining")
    
    # Display plant info
    st.header(f"Plant Information")
    st.write(current_row['plant_info'])
    
    if is_no_content:
        st.warning("This plant has no relevant content. Only validating article relevance scores.")
    
    # Load and display JSON data
    if article_relevance is not None:  # Always check article relevance
        form_submitted = False
        
        with st.form("validation_form"):
            article_validations = display_article_validation(search_results, article_relevance)
            
            if not is_no_content and all((search_results, scores)):
                score_validations = display_scores_validation(scores)
            else:
                score_validations = {}
            
            submitted = st.form_submit_button("Submit Validation")
        
        # Handle form submission
        if submitted and not st.session_state.form_submitted:
            st.session_state.form_submitted = True
            
            # Create validation output in the required format
            validation_output = create_validation_output(plant_code, article_validations, score_validations)
            
            # Save validation data
            if save_validation(plant_code, validation_output):
                st.success("Validation submitted successfully!")
                st.json(validation_output)  # Show the output format
                clear_form_state()
                form_submitted = True

        # Handle continue/auto-advance outside of form submission
        if form_submitted:
            if st.session_state.auto_advance:
                st.experimental_rerun()
            else:
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Next Project"):
                        st.experimental_rerun()

if __name__ == "__main__":
    main() 