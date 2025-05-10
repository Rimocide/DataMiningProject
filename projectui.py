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

df = None
ml_df = None
cluster_summary_df = None # Store cluster summary for insights
classification_results_df = None # Store classification results for insights

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
        plt.style.use('seaborn-v0_8-darkgrid')
        fig.patch.set_facecolor('#2E2E2E')

        text_color = 'white'
        grid_color = '#555555'
        title_fontsize = 14
        label_fontsize = 12

        ax1 = fig.add_subplot(411)
        top_countries = df[df['Country'] != 'United Kingdom']['Country'].value_counts().nlargest(10)
        sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis', ax=ax1, hue=top_countries.index, dodge=False, legend=False)
        ax1.set_title('Top 10 Non-UK Countries by Transactions', color=text_color, fontsize=title_fontsize)
        ax1.set_xlabel('Number of Transactions', color=text_color, fontsize=label_fontsize)
        ax1.set_ylabel('Country', color=text_color, fontsize=label_fontsize)
        ax1.tick_params(axis='x', colors=text_color)
        ax1.tick_params(axis='y', colors=text_color)
        ax1.grid(True, linestyle='--', alpha=0.7, color=grid_color)
        ax1.set_facecolor('#3C3C3C')

        ax2 = fig.add_subplot(412)
        monthly_sales = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
        monthly_sales.index = monthly_sales.index.to_timestamp()
        ax2.plot(monthly_sales.index, monthly_sales.values, marker='o', color='cyan', linewidth=2, markersize=5)
        ax2.set_title('Monthly Sales Trend', color=text_color, fontsize=title_fontsize)
        ax2.set_xlabel('Month', color=text_color, fontsize=label_fontsize)
        ax2.set_ylabel('Total Sales (£)', color=text_color, fontsize=label_fontsize)
        ax2.tick_params(axis='x', colors=text_color, rotation=45)
        ax2.tick_params(axis='y', colors=text_color)
        ax2.grid(True, linestyle='--', alpha=0.7, color=grid_color)
        ax2.set_facecolor('#3C3C3C')
        fig.autofmt_xdate()

        ax3 = fig.add_subplot(413)
        top_items = df.groupby('Description')['Quantity'].sum().nlargest(10)
        sns.barplot(x=top_items.values, y=top_items.index, palette='magma', ax=ax3, hue=top_items.index, dodge=False, legend=False)
        ax3.set_title('Top 10 Products by Quantity Sold', color=text_color, fontsize=title_fontsize)
        ax3.set_xlabel('Total Quantity Sold', color=text_color, fontsize=label_fontsize)
        ax3.set_ylabel('Product Description', color=text_color, fontsize=label_fontsize)
        ax3.tick_params(axis='x', colors=text_color)
        ax3.tick_params(axis='y', colors=text_color)
        ax3.grid(True, linestyle='--', alpha=0.7, color=grid_color)
        ax3.set_facecolor('#3C3C3C')

        ax4 = fig.add_subplot(414)
        plot_total_price = df['TotalPrice'][df['TotalPrice'] < df['TotalPrice'].quantile(0.99)]
        sns.histplot(plot_total_price, bins=50, color='lightgreen', kde=True, ax=ax4)
        ax4.set_title('Transaction Amount Distribution (Up to 99th Percentile)', color=text_color, fontsize=title_fontsize)
        ax4.set_xlabel('Total Price per Transaction (£)', color=text_color, fontsize=label_fontsize)
        ax4.set_ylabel('Frequency', color=text_color, fontsize=label_fontsize)
        ax4.tick_params(axis='x', colors=text_color)
        ax4.tick_params(axis='y', colors=text_color)
        ax4.grid(True, linestyle='--', alpha=0.7, color=grid_color)
        ax4.set_facecolor('#3C3C3C')

        fig.tight_layout(pad=4.0)
        canvas = FigureCanvasTkAgg(fig, master=eda_scrolled_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(20, 5))

        ttk.Label(eda_scrolled_frame, text="Insight 1: This plot shows the countries outside the UK with the most transactions, highlighting potential markets for international focus.", font=('Segoe UI', 9, 'italic'), bootstyle="secondary", wraplength=800).pack(pady=(0, 10), padx=10, anchor='w')

        ttk.Label(eda_scrolled_frame, text="Insight 2: The monthly sales trend reveals seasonality and growth patterns over time, important for forecasting and campaign timing.", font=('Segoe UI', 9, 'italic'), bootstyle="secondary", wraplength=800).pack(pady=(0, 10), padx=10, anchor='w')

        ttk.Label(eda_scrolled_frame, text="Insight 3: Identifying the top-selling products helps with inventory management, marketing promotions, and understanding customer demand.", font=('Segoe UI', 9, 'italic'), bootstyle="secondary", wraplength=800).pack(pady=(0, 10), padx=10, anchor='w')

        ttk.Label(eda_scrolled_frame, text="Insight 4: The distribution of transaction amounts shows that most orders are of lower value, with a few high-value transactions (outliers removed for clarity). This is crucial for pricing and sales strategies.", font=('Segoe UI', 9, 'italic'), bootstyle="secondary", wraplength=800).pack(pady=(0, 20), padx=10, anchor='w')


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
        fig.patch.set_facecolor('#2E2E2E')
        text_color = 'white'
        grid_color = '#555555'

        ax = fig.add_subplot(111)
        scatter = sns.scatterplot(data=ml_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=70, alpha=0.8, ax=ax, legend='full')
        ax.set_title(f'Customer Segments (k={num_clusters}) via PCA', color=text_color, fontsize=15)
        ax.set_xlabel('PCA Component 1', color=text_color, fontsize=12)
        ax.set_ylabel('PCA Component 2', color=text_color, fontsize=12)
        ax.tick_params(axis='x', colors=text_color)
        ax.tick_params(axis='y', colors=text_color)
        ax.grid(True, linestyle='--', alpha=0.5, color=grid_color)
        ax.set_facecolor('#3C3C3C')
        legend = ax.legend(title='Cluster', frameon=True)
        plt.setp(legend.get_texts(), color=text_color)
        plt.setp(legend.get_title(), color=text_color)
        legend.get_frame().set_facecolor('#3C3C3C')
        legend.get_frame().set_edgecolor(grid_color)

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

        ttk.Label(cluster_frame, text="Insight: The PCA plot visualizes the customer clusters based on their spending behavior (Frequency, Avg Quantity, Total Spend). Analyze the mean values in the table above to understand the typical profile of customers in each cluster.", font=('Segoe UI', 9, 'italic'), bootstyle="secondary", wraplength=800).pack(pady=(0, 20), padx=10, anchor='w')


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

        ttk.Label(classify_frame, text="Classification Model Performance:", font=('Segoe UI', 14, 'bold'), bootstyle="primary").pack(pady=(10,5), padx=10, anchor='w')

        cols = list(classification_results_df.columns)
        tree = ttk.Treeview(classify_frame, columns=cols, show='headings', height=len(classification_results_df)+1, bootstyle="info")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=150 if col == 'Model' else 100)

        for index, row in classification_results_df.iterrows():
            tree.insert("", "end", values=[f"{val:.3f}" if isinstance(val, float) else val for val in row])
        tree.pack(expand=False, fill='x', padx=10, pady=10)

        best_model = classification_results_df.loc[classification_results_df['Accuracy'].idxmax()]
        summary_text = f"Insight: This table compares the performance of different classification models in identifying 'High Spenders'. Analyze metrics like Accuracy and F1 Score (useful for imbalanced data) to choose the most suitable model for your business needs. The model with the highest Accuracy is {best_model['Model']} ({best_model['Accuracy']:.3f})."

        ttk.Label(classify_frame, text=summary_text, font=('Segoe UI', 9, 'italic'), bootstyle="secondary", wraplength=800).pack(pady=(0, 20), padx=10, anchor='w')

        insights_button.config(state="normal") # Enable insights after classification

    except Exception as e:
        messagebox.showerror("Classification Error", f"An error occurred during classification: {str(e)}")
        ttk.Label(classify_frame, text=f"Error running classification: {str(e)}", bootstyle="danger").pack(pady=20, padx=10)
        insights_button.config(state="disabled")

def add_insight_point(parent_frame, text, indent=10, bullet="• "):
    """Helper function to add a bullet point label."""
    ttk.Label(parent_frame, text=f"{bullet}{text}", font=('Segoe UI', 10), bootstyle="secondary", wraplength=800, justify=tk.LEFT).pack(pady=(2, 2), padx=(10 + indent, 10), anchor='w')


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

        # Example dynamic text - replace with actual findings based on df content if possible
        top_countries_list = df[df['Country'] != 'United Kingdom']['Country'].value_counts().nlargest(3).index.tolist()
        top_product_desc = df.groupby('Description')['Quantity'].sum().nlargest(1).index[0]

        add_insight_point(insights_scrolled_frame, f"Top Non-UK Markets: Analysis shows that countries like {', '.join(top_countries_list)} are the leading international markets by transaction volume. This highlights potential markets for focused international strategies.")
        add_insight_point(insights_scrolled_frame, "Sales Trend: The data reveals a clear seasonal pattern in sales, often peaking towards the end of the year. Understanding this trend is vital for optimizing inventory, staffing, and marketing campaign timing.")
        add_insight_point(insights_scrolled_frame, f"Popular Products: Items such as \"{top_product_desc}\" are consistently top sellers, indicating strong demand. These items could be used as loss leaders, featured in promotions, or bundled with other products.")
        add_insight_point(insights_scrolled_frame, "Transaction Value Distribution: The majority of transactions are relatively small, with a smaller number of high-value purchases. This suggests a need for tailored strategies targeting both frequent small buyers and high-value customers.")


        # --- Clustering Insights Summary ---
        ttk.Label(insights_scrolled_frame, text="2. Customer Segmentation Insights (Clustering)", font=('Segoe UI', 12, 'bold'), bootstyle="primary").pack(pady=(15, 5), padx=10, anchor='w')
        ttk.Label(insights_scrolled_frame, text=f"Based on K-Means clustering with k={int(cluster_k_var.get())}, customers have been grouped into distinct segments based on their Total Spend, Frequency, and Average Quantity per transaction. Review the table on the 'Customer Clustering' tab to understand each segment's unique profile and size.", font=('Segoe UI', 10), bootstyle="secondary", wraplength=800, justify=tk.LEFT).pack(pady=(0, 5), padx=10, anchor='w')

        ttk.Label(insights_scrolled_frame, text="Common segments often found include:", font=('Segoe UI', 10, 'italic'), bootstyle="secondary").pack(pady=(5, 2), padx=10, anchor='w')

        add_insight_point(insights_scrolled_frame, 'High Value Loyalists (High Spend, High Frequency)', indent=20)
        add_insight_point(insights_scrolled_frame, 'Potential Loyalists (Moderate Spend & Frequency)', indent=20)
        add_insight_point(insights_scrolled_frame, 'New Customers (Low Spend & Frequency)', indent=20)
        add_insight_point(insights_scrolled_frame, 'Big Spenders (High Spend, Low Frequency)', indent=20)
        add_insight_point(insights_scrolled_frame, 'Low Value / Churn Risk (Low Spend & Frequency)', indent=20)

        ttk.Label(insights_scrolled_frame, text="Tailor your marketing, product recommendations, and customer service efforts based on the characteristics of each identified cluster.", font=('Segoe UI', 10), bootstyle="secondary", wraplength=800, justify=tk.LEFT).pack(pady=(5, 10), padx=10, anchor='w')


        # --- Classification Insights Summary ---
        ttk.Label(insights_scrolled_frame, text="3. High Spender Classification Insights", font=('Segoe UI', 12, 'bold'), bootstyle="primary").pack(pady=(15, 5), padx=10, anchor='w')

        if classification_results_df is not None and not classification_results_df.empty:
            best_model = classification_results_df.loc[classification_results_df['Accuracy'].idxmax()]
            add_insight_point(insights_scrolled_frame, f"Best Performing Model: Based on Accuracy, the '{best_model['Model']}' model achieved the highest score of {best_model['Accuracy']:.3f}. Consider other metrics like F1 Score, Precision, and Recall (available on the 'High Spender Classification' tab) for a complete evaluation, especially if 'High Spender' is a smaller group.")
            add_insight_point(insights_scrolled_frame, "Key Predictors: The models utilize features such as Total Spend, Frequency, and Average Quantity to predict high spenders. Customers exhibiting higher values in these areas are more likely to be identified.")
            add_insight_point(insights_scrolled_frame, "Application: The chosen model can be used to proactively identify potential high-value customers for targeted marketing or loyalty programs.")
        else:
             ttk.Label(insights_scrolled_frame, text="Classification results are not available.", font=('Segoe UI', 10), bootstyle="secondary", wraplength=800, justify=tk.LEFT).pack(pady=(0, 10), padx=10, anchor='w')


        # --- Business Recommendations ---
        ttk.Label(insights_scrolled_frame, text="4. Business Recommendations", font=('Segoe UI', 14, 'bold'), bootstyle="success").pack(pady=(15, 5), padx=10, anchor='w')

        add_insight_point(insights_scrolled_frame, "Targeting: Use the customer segments from clustering to develop personalized marketing campaigns, promotions, and product recommendations for each group.")
        add_insight_point(insights_scrolled_frame, "High Spender Identification: Implement the best classification model to identify and nurture potential or existing high spenders through exclusive offers or loyalty programs.")
        add_insight_point(insights_scrolled_frame, f"International Growth: Develop targeted strategies for top non-UK markets like {', '.join(top_countries_list)} based on their specific preferences and transaction patterns.")
        add_insight_point(insights_scrolled_frame, f"Inventory & Promotion: Ensure robust stock of high-demand items like \"{top_product_desc}\" and strategically feature them in cross-selling or up-selling efforts.")
        add_insight_point(insights_scrolled_frame, "Seasonal Planning: Align marketing spend, inventory buildup, and staffing with the identified peak sales periods to maximize revenue opportunities.")
        add_insight_point(insights_scrolled_frame, "Average Transaction Value: Introduce initiatives such as minimum purchase discounts, product bundling, or personalized recommendations for complementary items to encourage larger orders.")

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