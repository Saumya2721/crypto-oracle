# Crypto Oracle: Predictive Market & Sentiment Dashboard

A full-stack financial application that implements a multivariate linear regression machine learning pipeline to forecast next-day cryptocurrency opening prices. By ingesting historical market constraints (OHLCV metrics) alongside contextual data, the application optimizes for accuracy while displaying interactive tracking analytics and corresponding real-time financial sentiment.

Built as an advanced Web Development Capstone Project.

## Key Architectural & Engineering Features
* **Machine Learning Engine:** Dynamically calculates predictive constraints utilizing a 4-feature Autoregressive (AR) structure via `ml-regression-multivariate-linear`. The pipeline optimizes weight values across daily Open, Volume, High, and Low parameters.
* **Accuracy Evaluation & Testing:** Computes Mean Absolute Error (MAE) sequentially on server initialization. The system performs comparative diagnostics between a compact 2-feature baseline (Open & Volume) and a 4-feature setup, proving that raw volatility metrics reduce structural tracking errors significantly.
* **Multi-Stream Data Pipeline:** Connects asynchronous REST API endpoints with Axios pipelines, using strict date sorting and key scrubbing to process raw JSON into clean computational matrices.
* **Context-Driven Search Ingestion:** Implements customized context-filtered queries via Boolean parameters (`AND crypto`) across dedicated media pipelines (The Guardian), restricting inputs to corporate and technology verticals to eradicate acronym search collisions.
* **Client-Side Rendering & UI:** Renders modular partial layouts using EJS templates. Securely shifts metrics to the presentation tier using HTML5 data attributes (`data-*`), decoupling computational arrays from script evaluation while eliminating Cross-Site Scripting (XSS) risks.
* **Interactive Charting Vector:** Renders highly fluid, tooltip-supported graphical history maps built directly with Chart.js line matrices.

## Tech Stack
* **Runtime Environment:** Node.js
* **Backend Framework:** Express.js
* **Data Processing & ML:** `ml-regression-multivariate-linear`, Axios, Dotenv
* **Frontend Matrix:** EJS (Embedded JavaScript Templates), Chart.js, Bootstrap 5, Native HTML5 Datalists

## Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Saumya2721/crypto-oracle.git](https://github.com/Saumya2721/crypto-oracle.git)
   cd crypto-oracle
2. **Load Global Dependencies:**
   ```bash
   npm install
3. **Configure the Environment:**
  *Create a .env configuration file in the project's root folder and provide your custom private keys:*
   ```env
  PORT=3000
  ALPHA_VANTAGE_KEY="YOUR_ALPHA_VANTAGE_DEVELOPER_KEY"
  NEWS_API_KEY="YOUR_GUARDIAN_API_KEY"

5. **Launch the Web App:**
   ```bash
   node index.js
