import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

class QuestionnaireAnalysis:
    # Define PANAS items (order is fixed)
    panas_items = [
        "Active", "Distressed", "Interested", "Inspired", "Annoyed", "Strong",
        "Guilty", "Scared", "Hostile", "Excited", "Proud", "Irritable",
        "Enthusiastic", "Ashamed", "Alert", "Nervous", "Determined", "Attentive",
        "Jittery", "Afraid", "Stressed", "Frustrated", "Happy", "Angry",
        "Irritated", "Sad"
    ]
    
    # Define STAI items (order is fixed)
    stai_items = [
        "I feel at ease", "I feel nervous", "I am jittery", "I am relaxed", 
        "I am worried", "I feel pleasant"
    ]
    
    # SAM items (Valence and Arousal)
    sam_items = ["Valence", "Arousal"]
    
    sssq_items = [
        "Committed to performance goals",
        "Wanted to succeed on the task",
        "Motivated to do the task",
        "Reflected about myself",
        "Worried about what others think",
        "Concerned about the impression I made"
    ]
    
    def __init__(self, file_path):
        # Initialize with a file path
        self.file_path = file_path
        self.df = None

    def parse_csv(self):
        """ Parse the CSV file and process PANAS, STAI, SAM, and SSSQ responses separately """
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        # Extract start times and phases from the relevant lines
        start_times = list(map(float, next(line for line in lines if line.startswith("# START")).strip().split(";")[1:6]))
        phases = next(line for line in lines if line.startswith("# ORDER")).strip().split(";")[1:6]

        # Extract PANAS responses
        panas_lines = [line.strip().split(";")[1:-1] for line in lines if line.startswith("# PANAS")]
        panas_responses = [
            [int(x) if x.isdigit() else np.nan for x in responses] 
            for responses in panas_lines
        ]

        # Handle 'Angry' and 'Irritated' for non-TSST phases by updating indices
        angry_idx, irritated_idx = self.panas_items.index("Angry"), self.panas_items.index("Irritated")
        for i, phase in enumerate(phases):
            if phase != 'TSST':
                panas_responses[i][angry_idx] = panas_responses[i][irritated_idx] = np.nan

        # Extract STAI responses
        stai_lines = [line for line in lines if line.startswith("# STAI")]
        stai_responses = [
            [int(x) if x.isdigit() else np.nan for x in line.strip().split(";")[1:-1]] 
            for line in stai_lines
        ]

        # Extract SAM responses (Valence and Arousal)
        sam_lines = [line for line in lines if line.startswith("# DIM")]
        sam_responses = [
            [int(x) if x.isdigit() else np.nan for x in line.strip().split(";")[1:3]] 
            for line in sam_lines
        ]

        # Extract SSSQ responses (Only one row for SSSQ)
        sssq_lines = [line for line in lines if line.startswith("# SSSQ")]
        sssq_responses = [
            [int(x) if x.isdigit() else np.nan for x in line.strip().split(";")[1:7]] 
            for line in sssq_lines
        ]

        # Check if there are any SSSQ responses and add to the dataframe separately
        sssq_data = sssq_responses[0] if sssq_responses else [np.nan] * len(self.sssq_items)

        # Create DataFrame for PANAS, STAI, and SAM
        data = []
        for phase, start, panas, stai, sam in zip(phases, start_times, panas_responses, stai_responses, sam_responses):
            row = {"Phase": phase, "Start (s)": start}
            row.update(dict(zip(self.panas_items, panas)))
            row.update(dict(zip(self.stai_items, stai)))
            row.update(dict(zip(self.sam_items, sam)))
            data.append(row)

        self.df = pd.DataFrame(data)

        # Ensure that all items for PANAS, STAI, and SAM are present
        all_items = self.panas_items + self.stai_items + self.sam_items
        for item in all_items:
            if item not in self.df.columns:
                self.df[item] = np.nan

        # Create a separate DataFrame for SSSQ responses (no phases)
        sssq_df = pd.DataFrame([sssq_data], columns=self.sssq_items)

        # Add the SSSQ data separately
        self.sssq_df = sssq_df

        return self.df

    def generate_PANAS_plot(self):
        """ Create and display the Plotly line plot for PANAS responses """
        df_long = self.df.melt(id_vars=["Phase", "Start (s)"], value_vars=self.panas_items, 
                               var_name="PANAS Item", value_name="Response")
        df_long = df_long.dropna(subset=["Response"])

        fig = px.line(df_long, x="Phase", y="Response", color="PANAS Item", 
                      title="PANAS Responses Across Phases", 
                      labels={"Response": "PANAS Response (1-5)", "Phase": "Phase"})

        fig.update_traces(mode='markers+lines', marker=dict(size=10, symbol='circle'))
        fig.show()

    def generate_STAI_plot(self):
        """ Create and display the Plotly line plot for STAI responses """
        df_long = self.df.melt(id_vars=["Phase", "Start (s)"], value_vars=self.stai_items, 
                               var_name="STAI Item", value_name="Response")
        df_long = df_long.dropna(subset=["Response"])

        fig = px.line(df_long, x="Phase", y="Response", color="STAI Item", 
                      title="STAI Responses Across Phases", 
                      labels={"Response": "STAI Response (1-4)", "Phase": "Phase"})

        fig.update_traces(mode='markers+lines', marker=dict(size=10, symbol='circle'))
        fig.show()

    def generate_SAM_plot(self):
        """ Create and display the Plotly line plot for SAM (Valence and Arousal) responses """
        df_long = self.df.melt(id_vars=["Phase", "Start (s)"], value_vars=self.sam_items, 
                               var_name="SAM Item", value_name="Response")
        df_long = df_long.dropna(subset=["Response"])

        fig = px.line(df_long, x="Phase", y="Response", color="SAM Item", 
                      title="SAM Responses Across Phases (Valence & Arousal)", 
                      labels={"Response": "SAM Response (1-9)", "Phase": "Phase"})

        fig.update_traces(mode='markers+lines', marker=dict(size=10, symbol='circle'))
        fig.show()
        

    def generate_SSSQ_plot(self):
        """ Generate an interactive bar chart for the SSSQ responses using Plotly (with flipped axes) """
        # Use the sssq_df for the plot
        sssq_data = self.sssq_df.iloc[0]  # Since there is only one row, just take the first row
        sssq_data = sssq_data.dropna()  # Drop NaN values if any

        # Create the bar chart using Plotly (flip axes by switching x and y)
        fig = go.Figure(data=[go.Bar(
            y=sssq_data.index,  # SSSQ items on the y-axis
            x=sssq_data.values,  # SSSQ scores on the x-axis
            orientation='h',  # Specify horizontal bars
            marker_color='blue'
        )])

        # Update layout
        fig.update_layout(
            title="SSSQ Responses",
            yaxis_title="SSSQ Items",
            xaxis_title="Scores",
            template="plotly_white"
        )

        # Show the plot
        fig.show()