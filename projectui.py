import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set dark theme globally for general stylistic elements,
# but we will explicitly set facecolors for figures/axes below
sns.set_theme(style="dark", rc={
    'axes.edgecolor': 'white',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#555555', # Keep original grid color preference
    'legend.facecolor': '#1e1e1e',
    'legend.edgecolor': '#555555',
    'legend.fontsize': 10,
    'legend.title_fontsize': 12,
}, font_scale=1.0)

# Define preferred facecolors
FIG_FACECOLOR = '#2E2E2E'
AXES_FACECOLOR = '#3C3C3C'
TEXT_COLOR = 'white' # Based on the theme setting

df = None
ml_df = None
cluster_summary_df = None
classification_results_df = None

def load_data():
    global df, ml_df, home_status_label, eda_button, cluster_button, classify_button, insights_button

    file_path = filedialog.askopenfilename(
        title="Select Online Retail CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file_path:
        home_status_label.config(text="File selection cancelled.", bootstyle="warning")
        return

    try:
        temp_df = pd.read_csv(file_path, encoding='ISO-8859-1')
        home_status_label.config(text=f"Loading {file_path.split('/')[-1]}...", bootstyle="info")
        root.update_idletasks()

        temp_df.dropna(subset=['CustomerID'], inplace=True)
        temp_df['CustomerID'] = temp_df['CustomerID'].astype(int)
        temp_df.drop_duplicates(inplace=True)
        temp_df = temp_df[(temp_df['Quantity'] > 0) & (temp_df['UnitPrice'] > 0)]

        temp_df['InvoiceDate'] = pd.to_datetime(temp_df['InvoiceDate'], errors='coerce')
        temp_df.dropna(subset=['InvoiceDate'], inplace=True)

        temp_df['TotalPrice'] = temp_df['Quantity'] * temp_df['UnitPrice']
        customer_spend = temp_df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
        customer_spend.columns = ['CustomerID', 'TotalSpend']

        median_spend = customer_spend['TotalSpend'].median()
        customer_spend['HighSpender'] = (customer_spend['TotalSpend'] > median_spend).astype(int)

        frequency = temp_df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index(name='Frequency')

        avg_quantity = temp_df.groupby('CustomerID')['Quantity'].mean().reset_index(name='AvgQuantity')

        temp_ml_df = customer_spend.merge(frequency, on='CustomerID').merge(avg_quantity, on='CustomerID')

        df = temp_df
        ml_df = temp_ml_df

        home_status_label.config(text=f"Data loaded successfully! Rows: {len(df)}, Columns: {len(df.columns)}\nML Features Ready: {len(ml_df)} customers", bootstyle="success")
        eda_button.config(state="normal")
        cluster_button.config(state="normal")
        classify_button.config(state="normal")
        insights_button.config(state="disabled")
        notebook.select(eda_tab)

    except FileNotFoundError:
        home_status_label.config(text="Error: File not found.", bootstyle="danger")
        messagebox.showerror("Error", "The specified file was not found.")
    except pd.errors.EmptyDataError:
        home_status_label.config(text="Error: The selected file is empty.", bootstyle="danger")
        messagebox.showerror("Error", "The selected CSV file is empty.")
    except pd.errors.ParserError:
        home_status_label.config(text="Error: Could not parse the CSV file. Please check format.", bootstyle="danger")
        messagebox.showerror("Error", "Could not parse the CSV file. Ensure it's a valid CSV.")
    except KeyError as e:
        home_status_label.config(text=f"Error: Missing expected column: {e}", bootstyle="danger")
        messagebox.showerror("Data Loading Error", f"The CSV file is missing an expected column: {e}. Please ensure columns like 'CustomerID', 'Quantity', 'UnitPrice', 'InvoiceNo', 'InvoiceDate' are present.")
    except Exception as e:
        home_status_label.config(text=f"An unexpected error occurred: {str(e)}", bootstyle="danger")
        messagebox.showerror("Error", f"An unexpected error occurred during data loading: {str(e)}")
        df = None
        ml_df = None
        eda_button.config(state="disabled")
        cluster_button.config(state="disabled")
        classify_button.config(state="disabled")
        insights_button.config(state="disabled")


def show_eda():
    global df
    if df is None:
        messagebox.showwarning("Data Not Loaded", "Please load the data first from the 'Load Data' tab.")
        return

    for widget in eda_scrolled_frame.winfo_children():
        widget.destroy()

    try:
        if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
             df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')

        fig = plt.Figure(figsize=(14, 16), dpi=100)
        fig.patch.set_facecolor(FIG_FACECOLOR) # Set figure facecolor

        title_fontsize = 14
        label_fontsize = 12

        ax1 = fig.add_subplot(411)
        ax1.set_facecolor(AXES_FACECOLOR) # Set axes facecolor
        top_countries = df[df['Country'] != 'United Kingdom']['Country'].value_counts().nlargest(10)
        sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis', ax=ax1, hue=top_countries.index, dodge=False, legend=False)
        ax1.set_title('Top 10 Non-UK Countries by Transactions', color=TEXT_COLOR, fontsize=title_fontsize)
        ax1.set_xlabel('Number of Transactions', color=TEXT_COLOR, fontsize=label_fontsize)
        ax1.set_ylabel('Country', color=TEXT_COLOR, fontsize=label_fontsize)
        ax1.tick_params(axis='x', colors=TEXT_COLOR)
        ax1.tick_params(axis='y', colors=TEXT_COLOR)
        # Grid color comes from sns.set_theme now
        ax1.grid(True, linestyle='--', alpha=0.7)


        ax2 = fig.add_subplot(412)
        ax2.set_facecolor(AXES_FACECOLOR) # Set axes facecolor
        monthly_sales = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
        monthly_sales.index = monthly_sales.index.to_timestamp()
        ax2.plot(monthly_sales.index, monthly_sales.values, marker='o', color='cyan', linewidth=2, markersize=5)
        ax2.set_title('Monthly Sales Trend', color=TEXT_COLOR, fontsize=title_fontsize)
        ax2.set_xlabel('Month', color=TEXT_COLOR, fontsize=label_fontsize)
        ax2.set_ylabel('Total Sales (£)', color=TEXT_COLOR, fontsize=label_fontsize)
        ax2.tick_params(axis='x', colors=TEXT_COLOR, rotation=45)
        ax2.tick_params(axis='y', colors=TEXT_COLOR)
        # Grid color comes from sns.set_theme now
        ax2.grid(True, linestyle='--', alpha=0.7)
        fig.autofmt_xdate()

        ax3 = fig.add_subplot(413)
        ax3.set_facecolor(AXES_FACECOLOR) # Set axes facecolor
        top_items = df.groupby('Description')['Quantity'].sum().nlargest(10)
        sns.barplot(x=top_items.values, y=top_items.index, palette='magma', ax=ax3, hue=top_items.index, dodge=False, legend=False)
        ax3.set_title('Top 10 Products by Quantity Sold', color=TEXT_COLOR, fontsize=title_fontsize)
        ax3.set_xlabel('Total Quantity Sold', color=TEXT_COLOR, fontsize=label_fontsize)
        ax3.set_ylabel('Product Description', color=TEXT_COLOR, fontsize=label_fontsize)
        ax3.tick_params(axis='x', colors=TEXT_COLOR)
        ax3.tick_params(axis='y', colors=TEXT_COLOR)
        # Grid color comes from sns.set_theme now
        ax3.grid(True, linestyle='--', alpha=0.7)

        ax4 = fig.add_subplot(414)
        ax4.set_facecolor(AXES_FACECOLOR) # Set axes facecolor
        plot_total_price = df['TotalPrice'][df['TotalPrice'] < df['TotalPrice'].quantile(0.99)]
        sns.histplot(plot_total_price, bins=50, color='lightgreen', kde=True, ax=ax4)
        ax4.set_title('Transaction Amount Distribution (Up to 99th Percentile)', color=TEXT_COLOR, fontsize=title_fontsize)
        ax4.set_xlabel('Total Price per Transaction (£)', color=TEXT_COLOR, fontsize=label_fontsize)
        ax4.set_ylabel('Frequency', color=TEXT_COLOR, fontsize=label_fontsize)
        ax4.tick_params(axis='x', colors=TEXT_COLOR)
        ax4.tick_params(axis='y', colors=TEXT_COLOR)
        # Grid color comes from sns.set_theme now
        ax4.grid(True, linestyle='--', alpha=0.7)

        fig.tight_layout(pad=4.0)
        canvas = FigureCanvasTkAgg(fig, master=eda_scrolled_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(20, 5))

        ttk.Label(eda_scrolled_frame, text="Insight 1: This plot shows the countries outside the UK with the most transactions, highlighting potential markets for international focus.", font=('Segoe UI', 9, 'italic'), bootstyle="light", wraplength=800).pack(pady=(0, 10), padx=10, anchor='w')

        ttk.Label(eda_scrolled_frame, text="Insight 2: The monthly sales trend reveals seasonality and growth patterns over time, important for forecasting and campaign timing.", font=('Segoe UI', 9, 'italic'), bootstyle="light", wraplength=800).pack(pady=(0, 10), padx=10, anchor='w')

        ttk.Label(eda_scrolled_frame, text="Insight 3: Identifying the top-selling products helps with inventory management, marketing promotions, and understanding customer demand.", font=('Segoe UI', 9, 'italic'), bootstyle="light", wraplength=800).pack(pady=(0, 10), padx=10, anchor='w')

        ttk.Label(eda_scrolled_frame, text="Insight 4: The distribution of transaction amounts shows that most orders are of lower value, with a few high-value transactions (outliers removed for clarity). This is crucial for pricing and sales strategies.", font=('Segoe UI', 9, 'italic'), bootstyle="light", wraplength=800).pack(pady=(0, 20), padx=10, anchor='w')


    except Exception as e:
        messagebox.showerror("EDA Error", f"An error occurred while generating EDA plots: {str(e)}")
        ttk.Label(eda_scrolled_frame, text=f"Error generating EDA: {str(e)}", bootstyle="danger").pack(pady=20, padx=10)


def run_clustering():
    global ml_df, cluster_summary_df
    if ml_df is None:
        messagebox.showwarning("Data Not Ready", "Please load data first. ML features (ml_df) are not available.")
        return

    for widget in cluster_frame.winfo_children():
        widget.destroy()

    try:
        features = ml_df[['Frequency', 'AvgQuantity', 'TotalSpend']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        num_clusters = int(cluster_k_var.get())
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(features_scaled)
        ml_df['Cluster'] = clusters

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(features_scaled)
        ml_df['PCA1'] = reduced_data[:, 0]
        ml_df['PCA2'] = reduced_data[:, 1]

        fig = plt.Figure(figsize=(10, 7), dpi=100)
        fig.patch.set_facecolor(FIG_FACECOLOR) # Set figure facecolor

        ax = fig.add_subplot(111)
        ax.set_facecolor(AXES_FACECOLOR) # Set axes facecolor
        scatter = sns.scatterplot(data=ml_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=70, alpha=0.8, ax=ax, legend='full')
        ax.set_title(f'Customer Segments (k={num_clusters}) via PCA', color=TEXT_COLOR, fontsize=15)
        ax.set_xlabel('PCA Component 1', color=TEXT_COLOR, fontsize=12)
        ax.set_ylabel('PCA Component 2', color=TEXT_COLOR, fontsize=12)
        ax.tick_params(axis='x', colors=TEXT_COLOR)
        ax.tick_params(axis='y', colors=TEXT_COLOR)
        # Grid color comes from sns.set_theme now
        ax.grid(True, linestyle='--', alpha=0.5)
        legend = ax.legend(title='Cluster', frameon=True)
        # Legend colors should now be handled by sns.set_theme and explicit facecolor if needed
        # legend.get_frame().set_facecolor('#3C3C3C') # Already set by AXES_FACECOLOR if legend is within axes bounds? Let's rely on sns theme for legend box colors.


        canvas = FigureCanvasTkAgg(fig, master=cluster_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        score = silhouette_score(features_scaled, clusters)
        ttk.Label(cluster_frame, text=f'Silhouette Score (k={num_clusters}): {score:.3f}', font=('Segoe UI', 12, 'bold'), bootstyle="info").pack(pady=10, padx=10, anchor='w')

        cluster_summary_df = ml_df.groupby('Cluster')[['TotalSpend', 'Frequency', 'AvgQuantity']].mean().round(2)
        summary_label = ttk.Label(cluster_frame, text="Cluster Characteristics (Mean Values):", font=('Segoe UI', 12, 'bold'), bootstyle="primary")
        summary_label.pack(pady=(10,5), padx=10, anchor='w')

        cols = list(cluster_summary_df.reset_index().columns)
        tree = ttk.Treeview(cluster_frame, columns=cols, show='headings', height=len(cluster_summary_df)+1, bootstyle="primary")
        for col_name in cols:
            tree.heading(col_name, text=col_name)
            tree.column(col_name, anchor="center", width=120)

        for i, row in cluster_summary_df.reset_index().iterrows():
            tree.insert("", "end", values=list(row))
        tree.pack(pady=5, padx=10, fill=tk.X)

        ttk.Label(cluster_frame, text="Insight: The PCA plot visualizes the customer clusters based on their spending behavior (Frequency, Avg Quantity, Total Spend). Analyze the mean values in the table above to understand the typical profile of customers in each cluster.", font=('Segoe UI', 9, 'italic'), bootstyle="light", wraplength=800).pack(pady=(0, 20), padx=10, anchor='w')


    except Exception as e:
        messagebox.showerror("Clustering Error", f"An error occurred during clustering: {str(e)}")
        ttk.Label(cluster_frame, text=f"Error running clustering: {str(e)}", bootstyle="danger").pack(pady=20, padx=10)

def run_classification():
    global ml_df, classification_results_df, insights_button
    if ml_df is None:
        messagebox.showwarning("Data Not Ready", "Please load data first. ML features (ml_df) are not available.")
        return

    for widget in classify_frame.winfo_children():
        if widget != classify_controls_frame:
             widget.destroy()

    try:
        X = ml_df[['TotalSpend', 'Frequency', 'AvgQuantity']]
        y = ml_df['HighSpender']

        if len(X) < 2 or len(y.unique()) < 2:
             ttk.Label(classify_frame, text="Not enough data or classes for classification.", font=('Segoe UI', 12), bootstyle="warning").pack(pady=20, padx=10)
             insights_button.config(state="disabled")
             return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y if len(y.unique()) > 1 else None)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5)
        }
        results_list = []

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            results_list.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(y_test, y_pred, zero_division=0)
            })

        classification_results_df = pd.DataFrame(results_list)

        ttk.Label(classify_frame, text="Classification Model Performance Metrics:", font=('Segoe UI', 14, 'bold'), bootstyle="primary").pack(pady=(10,5), padx=10, anchor='w')

        cols = list(classification_results_df.columns)
        tree = ttk.Treeview(classify_frame, columns=cols, show='headings', height=len(classification_results_df)+1, bootstyle="info")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=150 if col == 'Model' else 100)

        for index, row in classification_results_df.iterrows():
            tree.insert("", "end", values=[f"{val:.3f}" if isinstance(val, float) else val for val in row])
        tree.pack(expand=False, fill='x', padx=10, pady=10)

        ttk.Label(classify_frame, text="Insight: This table shows key performance metrics for each classification model.", font=('Segoe UI', 9, 'italic'), bootstyle="light", wraplength=800).pack(pady=(0, 10), padx=10, anchor='w')

        # --- Add the Classification Metrics Plot ---
        ttk.Label(classify_frame, text="Model Performance Comparison:", font=('Segoe UI', 14, 'bold'), bootstyle="primary").pack(pady=(10,5), padx=10, anchor='w')

        fig_metrics = plt.Figure(figsize=(12, 6), dpi=100)
        fig_metrics.patch.set_facecolor(FIG_FACECOLOR) # Set figure facecolor
        ax_metrics = fig_metrics.add_subplot(111)
        ax_metrics.set_facecolor(AXES_FACECOLOR) # Set axes facecolor


        melted = classification_results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

        sns.barplot(x='Model', y='Score', hue='Metric', data=melted, palette='crest', ax=ax_metrics)

        ax_metrics.set_title('Model Comparison by Performance Metric', color=TEXT_COLOR, fontsize=14)
        ax_metrics.set_ylabel('Score', color=TEXT_COLOR)
        ax_metrics.set_xlabel('Model', color=TEXT_COLOR) # Added x-axis label
        ax_metrics.set_ylim(0, 1.05)
        # Grid color comes from sns.set_theme now
        ax_metrics.grid(True, linestyle='--', alpha=0.4)
        ax_metrics.tick_params(axis='x', colors=TEXT_COLOR)
        ax_metrics.tick_params(axis='y', colors=TEXT_COLOR)
        ax_metrics.legend(title='Metric') # Ensure legend is added


        fig_metrics.tight_layout(pad=2.0)

        canvas_metrics = FigureCanvasTkAgg(fig_metrics, master=classify_frame)
        canvas_metrics.draw()
        canvas_metrics.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)

        best_model = classification_results_df.loc[classification_results_df['Accuracy'].idxmax()]
        plot_insight_text = f"Insight: This graph visually compares the Accuracy, Precision, Recall, and F1 Score across the different classification models. The model with the highest Accuracy is {best_model['Model']} ({best_model['Accuracy']:.3f}). Use this visual and the table above to assess which model best meets your specific goals (e.g., prioritizing high recall to find as many high spenders as possible)."
        ttk.Label(classify_frame, text=plot_insight_text, font=('Segoe UI', 9, 'italic'), bootstyle="light", wraplength=800).pack(pady=(0, 20), padx=10, anchor='w')
        # --- End of Plot ---

        insights_button.config(state="normal")

    except Exception as e:
        messagebox.showerror("Classification Error", f"An error occurred during classification: {str(e)}")
        ttk.Label(classify_frame, text=f"Error running classification: {str(e)}", bootstyle="danger").pack(pady=20, padx=10)
        insights_button.config(state="disabled")

def add_insight_point(parent_frame, text, indent=20, bullet="• ", bootstyle="light"):
    """Helper function to add a bullet point label with light bootstyle."""
    ttk.Label(parent_frame, text=f"{bullet}{text}", font=('Segoe UI', 10), bootstyle=bootstyle, wraplength=800, justify=tk.LEFT).pack(pady=(2, 2), padx=(10 + indent, 10), anchor='w')


def generate_insights():
    global df, ml_df, cluster_summary_df, classification_results_df
    if df is None or ml_df is None or cluster_summary_df is None or classification_results_df is None:
         messagebox.showwarning("Analysis Not Complete", "Please load data and run EDA, Clustering, and Classification first.")
         return

    for widget in insights_scrolled_frame.winfo_children():
        widget.destroy()

    try:
        ttk.Label(insights_scrolled_frame, text="Business Insights and Recommendations", font=('Segoe UI', 16, 'bold'), bootstyle="info").pack(pady=(10, 15), padx=10, anchor='w')

        # --- EDA Insights Summary ---
        ttk.Label(insights_scrolled_frame, text="1. Key Findings from Exploratory Data Analysis (EDA)", font=('Segoe UI', 12, 'bold'), bootstyle="primary").pack(pady=(10, 5), padx=10, anchor='w')

        top_countries_list = df[df['Country'] != 'United Kingdom']['Country'].value_counts().nlargest(3).index.tolist() if df is not None else ["loading..."]
        top_product_desc = df.groupby('Description')['Quantity'].sum().nlargest(1).index[0] if df is not None and not df.empty else "loading..."


        add_insight_point(insights_scrolled_frame, f"Top Non-UK Markets: Analysis shows that countries like {', '.join(top_countries_list)} are the leading international markets by transaction volume. This highlights potential markets for focused international strategies.")
        add_insight_point(insights_scrolled_frame, "Sales Trend: The data reveals a clear seasonal pattern in sales, often peaking towards the end of the year. Understanding this trend is vital for optimizing inventory, staffing, and marketing campaign timing.")
        add_insight_point(insights_scrolled_frame, f"Popular Products: Items such as \"{top_product_desc}\" are consistently top sellers, indicating strong demand. These items could be used as loss leaders, featured in promotions, or bundled with other products.")
        add_insight_point(insights_scrolled_frame, "Transaction Value Distribution: The majority of transactions are relatively small, with a smaller number of high-value purchases. This suggests a need for tailored strategies targeting both frequent small buyers and high-value customers.")


        # --- Clustering Insights Summary ---
        ttk.Label(insights_scrolled_frame, text="2. Customer Segmentation Insights (Clustering)", font=('Segoe UI', 12, 'bold'), bootstyle="primary").pack(pady=(15, 5), padx=10, anchor='w')
        ttk.Label(insights_scrolled_frame, text=f"Based on K-Means clustering with k={int(cluster_k_var.get())}, customers have been grouped into distinct segments based on their Total Spend, Frequency, and Average Quantity per transaction. Review the table and plot on the 'Customer Clustering' tab to understand each segment's unique profile and size.", font=('Segoe UI', 10), bootstyle="light", wraplength=800, justify=tk.LEFT).pack(pady=(0, 5), padx=10, anchor='w')

        ttk.Label(insights_scrolled_frame, text="Common segments often found include:", font=('Segoe UI', 10, 'italic'), bootstyle="light").pack(pady=(5, 2), padx=10, anchor='w')

        add_insight_point(insights_scrolled_frame, 'High Value Loyalists (Typically high spend, high frequency)', indent=20)
        add_insight_point(insights_scrolled_frame, 'Potential Loyalists (Often moderate spend and frequency, potential for growth)', indent=20)
        add_insight_point(insights_scrolled_frame, 'New Customers (Recently acquired, lower initial metrics)', indent=20)
        add_insight_point(insights_scrolled_frame, 'Big Spenders (High total spend but perhaps lower frequency)', indent=20)
        add_insight_point(insights_scrolled_frame, 'Low Value / Churn Risk (Low spend and frequency)', indent=20)

        ttk.Label(insights_scrolled_frame, text="Tailor your marketing, product recommendations, and customer service efforts based on the characteristics of each identified cluster to improve engagement and loyalty.", font=('Segoe UI', 10), bootstyle="light", wraplength=800, justify=tk.LEFT).pack(pady=(5, 10), padx=10, anchor='w')


        # --- Classification Insights Summary ---
        ttk.Label(insights_scrolled_frame, text="3. High Spender Classification Insights", font=('Segoe UI', 12, 'bold'), bootstyle="primary").pack(pady=(15, 5), padx=10, anchor='w')

        if classification_results_df is not None and not classification_results_df.empty:
            best_model = classification_results_df.loc[classification_results_df['Accuracy'].idxmax()]
            add_insight_point(insights_scrolled_frame, f"Best Performing Model: Based on Accuracy, the '{best_model['Model']}' model achieved the highest score of {best_model['Accuracy']:.3f}. Consider other metrics (F1 Score, Precision, Recall) and the performance comparison plot on the 'High Spender Classification' tab for a complete evaluation.")
            add_insight_point(insights_scrolled_frame, "Key Predictors: Features like Total Spend, Frequency, and Average Quantity are the primary indicators used by the models to predict high spenders.")
            add_insight_point(insights_scrolled_frame, "Application: The selected model provides a data-driven way to proactively identify customers likely to be high spenders, enabling targeted high-value customer strategies.")
        else:
             ttk.Label(insights_scrolled_frame, text="Classification results are not available.", font=('Segoe UI', 10), bootstyle="light", wraplength=800, justify=tk.LEFT).pack(pady=(0, 10), padx=10, anchor='w')


        # --- Business Recommendations ---
        ttk.Label(insights_scrolled_frame, text="4. Business Recommendations", font=('Segoe UI', 14, 'bold'), bootstyle="success").pack(pady=(15, 5), padx=10, anchor='w')

        add_insight_point(insights_scrolled_frame, "Targeting: Implement segmented marketing campaigns based on the customer clusters identified, personalizing offers and communications.")
        add_insight_point(insights_scrolled_frame, "High Spender Strategy: Utilize the classification model to identify potential and existing high spenders for exclusive loyalty programs, early access to products, or premium support.")
        add_insight_point(insights_scrolled_frame, f"International Expansion: Invest strategically in top non-UK markets ({', '.join(top_countries_list)}) by localizing offerings and marketing efforts.")
        add_insight_point(insights_scrolled_frame, f"Inventory & Promotion: Ensure consistent availability of top-selling products like \"{top_product_desc}\". Create product bundles or suggest complementary items during checkout to increase basket size.")
        add_insight_point(insights_scrolled_frame, "Seasonal Readiness: Use the historical sales trends to forecast demand, optimize inventory, and plan staffing levels for peak seasons.")
        add_insight_point(insights_scrolled_frame, "Average Transaction Value Growth: Introduce initiatives such as tiered discounts, free shipping thresholds, or personalized product recommendations for complementary items to encourage larger orders.")


    except Exception as e:
        messagebox.showerror("Insights Error", f"An error occurred while generating insights: {str(e)}")
        ttk.Label(insights_scrolled_frame, text=f"Error generating insights: {str(e)}", bootstyle="danger").pack(pady=20, padx=10)


root = tb.Window(themename="darkly")
root.title("Online Retail Customer Analysis Dashboard")
root.geometry("1200x900")

app_frame = tb.Frame(root, padding=10)
app_frame.pack(fill=BOTH, expand=YES)

notebook = tb.Notebook(app_frame, bootstyle="primary")
notebook.pack(fill=BOTH, expand=YES, padx=5, pady=5)

home_tab = tb.Frame(notebook, padding=20)
notebook.add(home_tab, text=" 1. Load Data ")

home_title = tb.Label(home_tab, text="Welcome to the Customer Analysis Dashboard", font=('Segoe UI', 18, 'bold'), bootstyle="info")
home_title.pack(pady=(10, 20))

home_instructions = tb.Label(home_tab, text="Please load your Online Retail CSV file to begin the analysis.", font=('Segoe UI', 12))
home_instructions.pack(pady=(0, 20))

load_button = tb.Button(home_tab, text="Browse & Load Dataset", bootstyle="success-outline", command=load_data, width=30)
load_button.pack(pady=20)

home_status_label = tb.Label(home_tab, text="No data loaded.", font=('Segoe UI', 10), bootstyle="secondary")
home_status_label.pack(pady=10, fill=tk.X)

tb.Separator(home_tab, bootstyle="secondary").pack(fill=tk.X, pady=20, padx=50)
tb.Label(home_tab, text="Expected CSV Columns: CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice, Country, Description (optional for some plots)", font=('Segoe UI', 9, 'italic'), bootstyle="light").pack(pady=5)

eda_tab = tb.Frame(notebook, padding=10)
notebook.add(eda_tab, text=" 2. Exploratory Analysis ")

eda_controls_frame = tb.Frame(eda_tab, padding=(0,0,0,10))
eda_controls_frame.pack(fill=tk.X)
eda_button = tb.Button(eda_controls_frame, text="📊 Generate EDA Visualizations", bootstyle="info-outline", command=show_eda, state="disabled")
eda_button.pack(pady=10)

eda_scrolled_frame = ScrolledFrame(eda_tab, autohide=True, bootstyle="round")
eda_scrolled_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

initial_eda_label = tb.Label(eda_scrolled_frame, text="Click 'Generate EDA Visualizations' after loading data.", font=('Segoe UI', 12), bootstyle="secondary")
initial_eda_label.pack(pady=50, padx=20)


cluster_tab = tb.Frame(notebook, padding=10)
notebook.add(cluster_tab, text=" 3. Customer Clustering ")

cluster_controls_frame = tb.Frame(cluster_tab, padding=(0,0,0,10))
cluster_controls_frame.pack(fill=tk.X)

k_frame = tb.Frame(cluster_controls_frame)
k_frame.pack(pady=5)
tb.Label(k_frame, text="Number of Clusters (K):", font=('Segoe UI', 10)).pack(side=LEFT, padx=(0,5))
cluster_k_var = tk.StringVar(value="3")
k_spinbox = tb.Spinbox(k_frame, from_=2, to=10, textvariable=cluster_k_var, width=5, bootstyle="info")
k_spinbox.pack(side=LEFT, padx=(0,10))

cluster_button = tb.Button(k_frame, text="🧩 Run Clustering Analysis", bootstyle="info-outline", command=run_clustering, state="disabled")
cluster_button.pack(side=LEFT, padx=10)

cluster_frame = tb.Frame(cluster_tab)
cluster_frame.pack(fill=tk.BOTH, expand=True)

initial_cluster_label = tb.Label(cluster_frame, text="Adjust K and click 'Run Clustering Analysis' after loading data.", font=('Segoe UI', 12), bootstyle="secondary")
initial_cluster_label.pack(pady=50, padx=20)

classify_tab_page = tb.Frame(notebook, padding=0)
notebook.add(classify_tab_page, text=" 4. High Spender Classification ")

classify_tab_scrolled = ScrolledFrame(classify_tab_page, autohide=True, bootstyle="round")
classify_tab_scrolled.pack(fill=BOTH, expand=YES, padx=5, pady=5)

classify_frame = classify_tab_scrolled

classify_controls_frame = tb.Frame(classify_frame, padding=(0,0,0,10))
classify_controls_frame.pack(fill=tk.X, pady=(5,0), padx=5)
classify_button = tb.Button(classify_controls_frame, text="🎯 Run Classification Models", bootstyle="info-outline", command=run_classification, state="disabled")
classify_button.pack(pady=10)

initial_classify_label = tb.Label(classify_frame, text="Click 'Run Classification Models' after loading data.", font=('Segoe UI', 12), bootstyle="secondary")
initial_classify_label.pack(pady=50, padx=20)


insights_tab = tb.Frame(notebook, padding=10)
notebook.add(insights_tab, text=" 5. Business Insights ")

insights_controls_frame = tb.Frame(insights_tab, padding=(0,0,0,10))
insights_controls_frame.pack(fill=tk.X)
insights_button = tb.Button(insights_controls_frame, text="💡 Generate Business Insights", bootstyle="success-outline", command=generate_insights, state="disabled")
insights_button.pack(pady=10)

insights_scrolled_frame = ScrolledFrame(insights_tab, autohide=True, bootstyle="round")
insights_scrolled_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

initial_insights_label = tb.Label(insights_scrolled_frame, text="Run the analysis steps (EDA, Clustering, Classification) first to generate insights.", font=('Segoe UI', 12), bootstyle="secondary")
initial_insights_label.pack(pady=50, padx=20)


root.mainloop()