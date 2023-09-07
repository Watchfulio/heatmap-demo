import asyncio
import gc
import logging
import os
from collections import Counter
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import scipy
import seaborn as sns
import streamlit as st
import tiktoken
import spacy_alignments as tokenizations
import torch
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, kendalltau, gaussian_kde
from scipy.spatial.distance import euclidean
from tenacity import retry
from transformers import GPT2LMHeadModel


openai.api_key = os.getenv("OPENAI_API_KEY", "Not found")
openai.organization = os.getenv("OPENAI_ORGANIZATION", "Not found")


def render_heatmap(original_text, importance_scores_df):
    # Extract the importance scores
    importance_values = importance_scores_df["importance_value"].values

    # Check for division by zero during normalization
    min_val = np.min(importance_values)
    max_val = np.max(importance_values)

    if max_val - min_val != 0:
        normalized_importance_values = (importance_values - min_val) / (
            max_val - min_val
        )
    else:
        normalized_importance_values = np.zeros_like(
            importance_values
        )  # Fallback: all-zero array

    # Generate a colormap for the heatmap
    cmap = matplotlib.colormaps["inferno"]

    # Function to determine text color based on background color
    def get_text_color(bg_color):
        brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        if brightness < 0.5:
            return "white"
        else:
            return "black"

    # Initialize pointers for the original text and token importance
    original_pointer = 0
    token_pointer = 0

    # Create an HTML representation
    html = ""
    while original_pointer < len(original_text):
        token = importance_scores_df.loc[token_pointer, "token"]
        if original_pointer == original_text.find(token, original_pointer):
            importance = normalized_importance_values[token_pointer]
            rgba = cmap(importance)
            bg_color = rgba[:3]
            text_color = get_text_color(bg_color)
            html += f'<span style="background-color: rgba({int(bg_color[0]*255)}, {int(bg_color[1]*255)}, {int(bg_color[2]*255)}, 1); color: {text_color};">{token}</span>'
            original_pointer += len(token)
            token_pointer += 1
        else:
            html += original_text[original_pointer]
            original_pointer += 1

    # Display using Streamlit
    st.markdown(html, unsafe_allow_html=True)


def align_dataframes(b2a, df1, a2b, df2):
    aligned_strs1 = []
    aligned_vals1 = []
    aligned_strs2 = []
    aligned_vals2 = []

    # Keep track of seen indices to avoid duplication
    seen_indices1 = set()
    seen_indices2 = set()

    # Align df1 to df2
    for x in b2a:
        aligned_str1 = ""
        aligned_sum1 = 0
        aligned_ct1 = 0
        for y in x:
            if y not in seen_indices1:
                entry = df1.iloc[y]
                aligned_str1 += entry["token"]
                aligned_sum1 += entry["importance_value"]
                aligned_ct1 += 1
                seen_indices1.add(y)

        if aligned_ct1 > 0:
            aligned_strs1.append(aligned_str1)
            aligned_vals1.append(aligned_sum1 / aligned_ct1)

    # Align df2 to df1
    for x in a2b:
        aligned_str2 = ""
        aligned_sum2 = 0
        aligned_ct2 = 0
        for y in x:
            if y not in seen_indices2:
                entry = df2.iloc[y]
                aligned_str2 += entry["token"]
                aligned_sum2 += entry["importance_value"]
                aligned_ct2 += 1
                seen_indices2.add(y)

        if aligned_ct2 > 0:
            aligned_strs2.append(aligned_str2)
            aligned_vals2.append(aligned_sum2 / aligned_ct2)

    # Create aligned dataframes
    aligned_df1 = pd.DataFrame(
        {"token": aligned_strs1, "importance_value": aligned_vals1}
    )
    aligned_df2 = pd.DataFrame(
        {"token": aligned_strs2, "importance_value": aligned_vals2}
    )

    # Ensure both dataframes have identical rows and columns
    if len(aligned_df1) < len(aligned_df2):
        padding = len(aligned_df2) - len(aligned_df1)
        aligned_df1 = pd.concat(
            [
                aligned_df1,
                pd.DataFrame(
                    {"token": [None] * padding, "importance_value": [0.0] * padding}
                ),
            ]
        ).reset_index(drop=True)
    elif len(aligned_df1) > len(aligned_df2):
        padding = len(aligned_df1) - len(aligned_df2)
        aligned_df2 = pd.concat(
            [
                aligned_df2,
                pd.DataFrame(
                    {"token": [None] * padding, "importance_value": [0.0] * padding}
                ),
            ]
        ).reset_index(drop=True)

    return aligned_df1, aligned_df2


def analyze_heatmap(df_input):
    df = df_input.copy()

    if "token" not in df.columns or "importance_value" not in df.columns:
        raise ValueError(
            "The DataFrame must contain 'token' and 'importance_value' columns."
        )

    df["Position"] = range(len(df))

    st.write("## Distribution of Importance Scores")
    # Calculate histogram data
    hist, bin_edges = np.histogram(df["importance_value"], bins=20)
    # Get the viridis colormap
    viridis = matplotlib.colormaps["viridis"]
    # Initialize the figure
    fig = go.Figure()
    # Create the histogram bars with viridis coloring
    for i, freq in enumerate(hist):
        color = f"rgb({int(viridis(i / (len(bin_edges) - 1))[0] * 255)}, {int(viridis(i / (len(bin_edges) - 1))[1] * 255)}, {int(viridis(i / (len(bin_edges) - 1))[2] * 255)})"
        fig.add_trace(
            go.Bar(
                x=[(bin_edges[i] + bin_edges[i + 1]) / 2],
                y=[freq],
                width=np.diff(bin_edges)[i],
                marker=dict(color=color),
            )
        )
    # Calculate and add the KDE line
    x_kde = np.linspace(min(df["importance_value"]), max(df["importance_value"]), 500)
    kde = gaussian_kde(df["importance_value"])
    y_kde = kde(x_kde) * sum(hist) * (bin_edges[1] - bin_edges[0])
    fig.add_trace(
        go.Scatter(
            x=x_kde, y=y_kde, mode="lines", line_shape="spline", line=dict(color="red")
        )
    )
    # Additional styling
    fig.update_layout(
        title="Distribution of Importance Scores",
        xaxis_title="Importance Value",
        yaxis_title="Frequency",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("## Importance Score per Token")
    # Normalize the importance values
    min_val = df["importance_value"].min()
    max_val = df["importance_value"].max()
    normalized_values = (df["importance_value"] - min_val) / (max_val - min_val)
    # Initialize the figure
    fig = go.Figure()
    # Create the bars, colored based on normalized importance_value
    for i, (token, norm_value) in enumerate(zip(df["token"], normalized_values)):
        color = f"rgb({int(viridis(norm_value)[0] * 255)}, {int(viridis(norm_value)[1] * 255)}, {int(viridis(norm_value)[2] * 255)})"
        fig.add_trace(
            go.Bar(
                x=[i],  # Use index for x-axis
                y=[df["importance_value"].iloc[i]],
                width=1.0,  # Set the width to make bars touch each other
                marker=dict(color=color),
            )
        )
    # Additional styling
    fig.update_layout(
        title="Importance Score per Token",
        xaxis_title="Token",
        yaxis_title="Importance Value",
        showlegend=False,
        bargap=0,  # Remove gap between bars
        xaxis=dict(  # Set tick labels to tokens
            tickmode="array",
            tickvals=list(range(len(df["token"]))),
            ticktext=list(df["token"]),
        ),
    )
    # Rotate x-axis labels by 45 degrees
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.write("## Top 10 Most Important Words")
    top_10_important = df.nlargest(10, "importance_value")
    st.write(top_10_important[["token", "importance_value"]])

    st.write("## Top 10 Least Important Words")
    top_10_least_important = df.nsmallest(10, "importance_value")
    st.write(top_10_least_important[["token", "importance_value"]])

    correlation, p_value = scipy.stats.pearsonr(df["importance_value"], df["Position"])
    st.write(
        f"## Correlation between importance and position in text: {correlation:.2f}"
    )
    st.write(f"P-value: {p_value:.2f}")

    st.write("## Correlation Plot")
    fig = px.scatter(
        df,
        x="Position",
        y="importance_value",
        trendline="lowess",
        title="Correlation between Importance and Position in Text",
    )
    st.plotly_chart(fig, use_container_width=True)


def compare_heatmaps(df1, df2):
    # Ensure the DataFrames have the required columns
    if (
        "token" not in df1.columns
        or "importance_value" not in df1.columns
        or "token" not in df2.columns
        or "importance_value" not in df2.columns
    ):
        raise ValueError(
            "Both DataFrames must contain 'token' and 'importance_value' columns."
        )

    # Replace None with ""
    df1["token"].fillna("", inplace=True)
    df2["token"].fillna("", inplace=True)

    # Extracting importance scores
    importance_scores1 = df1["importance_value"].values
    importance_scores2 = df2["importance_value"].values

    # Check length of both DataFrames
    if len(importance_scores1) != len(importance_scores2):
        raise ValueError("Both DataFrames must have the same length.")

    n = len(importance_scores1)  # Number of dimensions (tokens)

    # Calculating Pearson correlation coefficient
    correlation, _ = pearsonr(importance_scores1, importance_scores2)

    # Calculating Cosine similarity
    cos_similarity = cosine_similarity(
        importance_scores1.reshape(1, -1), importance_scores2.reshape(1, -1)
    )[0][0]

    # Calculating Euclidean distance
    euclid_distance = euclidean(importance_scores1, importance_scores2)

    # Calculating Normalized Euclidean distance
    normalized_euclid_distance = euclid_distance / np.sqrt(n)

    # Calculating Kendall's Tau
    tau, _ = kendalltau(importance_scores1, importance_scores2)

    # Creating a dictionary to store the results
    results = {
        "Pearson Correlation": correlation,
        "Cosine Similarity": cos_similarity,
        "Euclidean Distance": normalized_euclid_distance,
        "Kendall's Tau": tau,
    }

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        # Plotting the importance-by-token for DataFrame 1
        st.write("## Importance by Token for GPT3 Estimation")
        fig1 = px.bar(
            df1,
            x="token",
            y="importance_value",
            title="Importance by Token",
            color_discrete_sequence=["blue"],
        )
        st.plotly_chart(fig1)

    with col2:
        # Plotting the importance-by-token for DataFrame 2
        st.write("## Importance by Token for GPT2 Integrated Gradients")
        fig2 = px.bar(
            df2,
            x="token",
            y="importance_value",
            title="Importance by Token",
            color_discrete_sequence=["red"],
        )
        st.plotly_chart(fig2)

    # Display the comparison metrics
    st.write("## Comparison Metrics")
    st.json(results)
    return results


def decoded_tokens(string, tokenizer):
    return [tokenizer.decode([x]) for x in tokenizer.encode(string)]


def scale_importance_log(
    importance_scores,
    base=None,
    offset=0.0,
    min_percentile=0,
    max_percentile=100,
    smoothing_constant=1e-10,
    scaling_factor=1.0,
    bias=0.0,
):
    # Extract the importance values
    importance_values = np.array([score[1] for score in importance_scores])

    # Apply optional percentile-based clipping
    if min_percentile > 0 or max_percentile < 100:
        min_val = np.percentile(importance_values, min_percentile)
        max_val = np.percentile(importance_values, max_percentile)
        importance_values = np.clip(importance_values, min_val, max_val)

    # Subtract the minimum value and add the optional offset
    importance_values = importance_values - np.min(importance_values) + offset

    # Add smoothing constant to ensure non-zero values
    importance_values += smoothing_constant

    # Apply logarithmic scaling, with an optional base
    scaled_values = (
        np.log(importance_values)
        if base is None
        else np.log(importance_values) / np.log(base)
    )

    # Apply scaling factor and bias
    scaled_values = scaling_factor * scaled_values + bias

    # Normalize to the range [0, 1]
    scaled_values = (scaled_values - np.min(scaled_values)) / (
        np.max(scaled_values) - np.min(scaled_values)
    )

    # Pair the scaled values with the original tokens
    scaled_importance_scores = [
        (token, scaled_value)
        for token, scaled_value in zip(
            [score[0] for score in importance_scores], scaled_values
        )
    ]

    return scaled_importance_scores


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


@retry
async def get_embedding(input_text, model=None, tokenizer=None):
    if not model:
        resp = await openai.Embedding.acreate(
            input=input_text, model="text-embedding-ada-002"
        )
        return np.array(resp["data"][0]["embedding"])
    else:
        if not tokenizer:
            raise ValueError(
                "If 'model' is specified, 'tokenizer' must also be specified."
            )

        # Generate input_ids
        input_ids = torch.tensor([tokenizer.encode(input_text)])

        # Generate hidden states
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # Select multiple layers (in this case, all of them)
        selected_layers = outputs.hidden_states

        # Stack selected layers
        stacked_hidden_states = torch.stack(selected_layers)

        # Calculate the mean across layers (result will have same shape as single-layer hidden states)
        mean_hidden_states = torch.mean(stacked_hidden_states, dim=0)[0]

        # Count token frequencies
        token_freqs = Counter(tokenizer.encode(input_text))

        # Calculate weights as sqrt(inv_freq)
        inv_freqs = {k: 1 / v for k, v in token_freqs.items()}
        weights = [sqrt(inv_freqs[id.item()]) for id in input_ids[0]]

        # Normalize weights
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]

        # Compute weighted mean of hidden states
        weighted_embedding = sum(
            weight * hidden_state
            for weight, hidden_state in zip(weights, mean_hidden_states)
        )

        # Convert to numpy array and normalize
        weighted_embedding = weighted_embedding.detach().numpy()
        return normalize_vector(weighted_embedding)


async def approximate_importance(
    perturbed_text,
    original_embedding,
    progress_bar,
    progress_percent,
    st_column,
    model=None,
    tokenizer=None,
):
    perturbed_embedding = await get_embedding(perturbed_text, model, tokenizer)
    # Compute "importance" measures
    cosine_dist = (
        1
        - cosine_similarity(
            original_embedding.reshape(1, -1), perturbed_embedding.reshape(1, -1)
        )[0][0]
    )
    column_progress[st_column] += progress_percent
    progress_bar.progress(column_progress[st_column])
    return cosine_dist


async def ablated_relative_importance(
    input_text, tokenizer, progress_bar, st_column, model=None
):
    original_embedding = await get_embedding(input_text, model, tokenizer)
    tokens = decoded_tokens(input_text, tokenizer)
    importance_scores = []

    # Prepare the tasks
    tasks = []
    # This increment will pretty much never hit 100, but that's ok
    progress_increment = int(1 / len(tokens) * 100)
    for i in range(len(tokens)):
        perturbed_text = "".join(tokens[:i] + tokens[i + 1 :])
        task = approximate_importance(
            perturbed_text,
            original_embedding,
            progress_bar,
            progress_increment,
            st_column,
            model,
            tokenizer,
        )
        tasks.append(task)

    # Run tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Update importance_scores and progress bar
    for i, importance in enumerate(results):
        importance_scores.append((tokens[i], importance))

    column_progress[st_column] = 0
    return importance_scores


async def process_importance(importance_function, st_column, *args, **kwargs):
    # Reserve a placeholder for the progress bar
    progress_bar_placeholder = st.empty()

    # Initialize the progress bar in Streamlit
    progress_bar = progress_bar_placeholder.progress(0)

    importance_map = await importance_function(
        progress_bar=progress_bar, st_column=st_column, *args, **kwargs
    )

    importance_map_df = pd.DataFrame(
        importance_map, columns=["token", "importance_value"]
    )

    offset = importance_map_df["importance_value"].mean()

    importance_log = scale_importance_log(
        importance_map,
        base=None,
        offset=offset,
        min_percentile=0,
        max_percentile=100,
        scaling_factor=1,
        bias=0,
    )
    importance_log_df = pd.DataFrame(
        importance_log, columns=["token", "importance_value"]
    )

    progress_bar_placeholder.empty()

    del importance_map
    del importance_map_df
    gc.collect()
    return importance_log_df


def integrated_gradients(input_ids, baseline, model, progress_bar, n_steps=100):
    # Convert input_ids and baseline to LongTensors
    input_ids, baseline = input_ids.long(), baseline.long()

    # Initialize tensor to store accumulated gradients
    accumulated_grads = None

    # Create interpolated inputs
    alphas = torch.linspace(0, 1, n_steps)
    delta = input_ids - baseline

    # Initialize tqdm progress bar
    progress_increment = int(1 / n_steps * 100)
    progress = 0

    for alpha in alphas:
        # Update tqdm progress bar
        progress += progress_increment
        progress_bar.progress(progress)

        # In-place modification for memory efficiency
        interpolate = baseline + (alpha * delta).long()

        # Convert interpolated samples to embeddings
        interpolate_embedding = (
            model.transformer.wte(interpolate).clone().detach().requires_grad_(True)
        )

        # Forward pass
        output = model(inputs_embeds=interpolate_embedding, output_attentions=False)[0]

        # Aggregate the logits across all positions (using sum in this example)
        aggregated_logit = output.sum()

        # Backward pass to calculate gradients
        aggregated_logit.backward()

        # In-place addition to save memory
        if accumulated_grads is None:
            accumulated_grads = interpolate_embedding.grad.clone()
        else:
            accumulated_grads += interpolate_embedding.grad

        # Clear gradients
        model.zero_grad()
        interpolate_embedding.grad.zero_()

        # Explicitly free up memory
        del interpolate
        del interpolate_embedding
        del output
        del aggregated_logit
        gc.collect()

    # Compute average gradients
    avg_grads = accumulated_grads / n_steps

    # Compute attributions
    with torch.no_grad():
        input_embedding = model.transformer.wte(input_ids)
        baseline_embedding = model.transformer.wte(baseline)
        attributions = (input_embedding - baseline_embedding) * avg_grads

    return attributions


def process_integrated_gradients(input_text, gpt2tokenizer, model, n_steps=100):
    # Reserve a placeholder for the progress bar
    progress_bar_placeholder = st.empty()

    # Initialize the progress bar in Streamlit
    progress_bar = progress_bar_placeholder.progress(0)

    inputs = torch.tensor([gpt2tokenizer.encode(input_text)])

    gpt2tokens = decoded_tokens(input_text, gpt2tokenizer)

    # Initialize a baseline as zero tensor
    baseline = torch.zeros_like(inputs).long()

    # Compute Integrated Gradients targeting the aggregated sequence output
    attributions = integrated_gradients(inputs, baseline, model, progress_bar, n_steps)

    # Sum across the embedding dimensions to get a single attribution score per token
    attributions_sum = attributions.sum(axis=2).squeeze(0).detach().numpy()

    l2_norm_attributions = np.linalg.norm(attributions_sum, 2)
    normalized_attributions_sum = attributions_sum / l2_norm_attributions

    clamped_attributions_sum = np.where(
        normalized_attributions_sum < 0, 0, normalized_attributions_sum
    )

    progress_bar_placeholder.empty()

    attribution_df = pd.DataFrame(
        {"token": gpt2tokens, "importance_value": clamped_attributions_sum}
    )
    return attribution_df

#
# MAIN EXECUTION 
#

# Configure root logger to capture only WARN or higher level logs
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configure logger to capture INFO-level logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

st.set_page_config(layout="wide")


@st.cache_resource
def load_model(model_version):
    return GPT2LMHeadModel.from_pretrained(model_version, output_attentions=True)


@st.cache_resource
def load_tokenizer(tokenizer_name):
    return tiktoken.get_encoding(tokenizer_name)


column_progress = {1: 0, 2: 0}
model_type = "gpt2"
model_version = "gpt2"
model = load_model(model_version)
gpt2tokenizer = load_tokenizer("gpt2")
gpt3tokenizer = load_tokenizer("cl100k_base")


st.title("A surprisingly effective way to estimate token importances in LLM prompts")

# Create empty spaces for vertical centering
st.empty()
st.empty()
st.empty()

# Text box for prompt
user_input = st.text_area("Enter your prompt", "")

ig = st.toggle(
    "Compute GPT-2 Integrated Gradients (may take a while if page is under load)"
)

# Submit button right below the text box
if st.button("Submit"):
    logger.debug(f"PROMPT: {user_input}")
    if ig:
        logger.debug(f"Processing With Integrated Gradients")
        tab1, tab2 = st.tabs(["Individual Analysis", "Comparative Analysis"])
        df1 = []
        df2 = []
        with tab1:
            # Create two columns for heatmaps
            col1, col2 = st.columns(2)

            # Place heatmap in the first column
            with col1:
                st.header("Importance Estimation (GPT-3 Embeddings, Log Scaled)")
                importance_map_log_df = asyncio.run(
                    process_importance(
                        ablated_relative_importance, 1, user_input, gpt3tokenizer
                    )
                )
                render_heatmap(user_input, importance_map_log_df)
                analyze_heatmap(importance_map_log_df)
                df1 = importance_map_log_df

            # Place heatmap in the second column
            with col2:
                st.header("Importance 'Ground Truth' (GPT-2 Integrated Gradients)")
                attribution_df = process_integrated_gradients(
                    user_input, gpt2tokenizer, model, n_steps=100
                )
                render_heatmap(user_input, attribution_df)
                analyze_heatmap(attribution_df)
                df2 = attribution_df

        with tab2:
            with st.spinner("Computing, please wait..."):
                a2b, b2a = tokenizations.get_alignments(
                    df1["token"].values.tolist(), df2["token"].values.tolist()
                )
                aligned_df1, aligned_df2 = align_dataframes(b2a, df1, a2b, df2)
                results = compare_heatmaps(aligned_df1, aligned_df2)
                logger.debug(f"PROMPT RESULTS:\nPROMPT: {user_input}\nRESULTS: {results}")
    if not ig:
        st.header("Importance Estimation (GPT-3 Embeddings, Log Scaled)")
        importance_map_log_df = asyncio.run(
            process_importance(
                ablated_relative_importance, 1, user_input, gpt3tokenizer
            )
        )
        render_heatmap(user_input, importance_map_log_df)
        analyze_heatmap(importance_map_log_df)

# Create empty spaces for vertical centering
st.empty()
st.empty()
st.empty()
