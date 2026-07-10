import express from "express";
import axios from "axios";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const SUPPORTED_COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LINK", "DOT"];

app.post("/analyze", async (req, res) => {
    const cryptocurrency = req.body.cryptocurrency;
    if (!cryptocurrency || !SUPPORTED_COINS.includes(cryptocurrency.toUpperCase())) {
        return res.status(400).json({ error: "Unsupported or missing cryptocurrency" });
    }
    const market = "USD";

    try {
        console.log(`Fetching data for ${cryptocurrency}...`);
        
        // 1. Fetch Binance Data (up to ~5.5 years = 2000 days)
        // Binance limits 1000 klins per request
        let allKlines = [];
        let endTime = Date.now();
        
        for(let i=0; i<2; i++) {
            const url = `https://api.binance.com/api/v3/klines?symbol=${cryptocurrency}USDT&interval=1d&limit=1000&endTime=${endTime}`;
            const res = await axios.get(url).catch(err => null);
            if(!res || !res.data || res.data.length === 0) break;
            
            allKlines = [...res.data, ...allKlines];
            endTime = res.data[0][0] - 1; 
        }

        if(allKlines.length === 0) {
            throw new Error(`Could not fetch Binance data for ${cryptocurrency}USDT. It may not be listed.`);
        }

        // 2. Combine Data for Charts
        const chartDates = [];
        const chartPrices = [];
        
        for (const kline of allKlines) {
            const dateObj = new Date(kline[0]);
            const dateStr = dateObj.toISOString().split('T')[0];
            const close = parseFloat(kline[4]);
            
            chartDates.push(dateStr);
            chartPrices.push(close);
        }

        // 4. Fetch News Data
        const newsDataURL = `https://newsdata.io/api/1/crypto?apikey=${process.env.NEWS_API_KEY}&language=en&coin=${cryptocurrency.toLowerCase()}&prioritydomain=top`;
        let newsResponse = await axios.get(newsDataURL);
        let rawArticles = newsResponse.data.results || [];
        if (rawArticles.length === 0) {
            console.log("No coin-specific news found, fetching general crypto news...");
            const fallbackURL = `https://newsdata.io/api/1/crypto?apikey=${process.env.NEWS_API_KEY}&language=en&prioritydomain=top`;
            const fallbackResponse = await axios.get(fallbackURL);
            rawArticles = fallbackResponse.data.results || [];
        }
        const articles = rawArticles.slice(0, 5).map(article => ({
            title: article.title,
            description: article.description,
            url: article.link
        }));

        // 4. Send data to Python ML Server for prediction
        let predictionResult = { prediction: "ERROR", confidence: 0 };
        try {
            console.log("Sending data to ML server...");
            const mlServerUrl = process.env.ML_SERVER_URL || 'http://localhost:5000';
            const mlResponse = await axios.post(`${mlServerUrl}/predict`, { coin: cryptocurrency });
            predictionResult = mlResponse.data;
        } catch(err) {
            console.error("ML Server Error:", err.response?.data || err.message);
        }

        // We slice to match the 6 months (180 days) of data at max for the chart
        const displayDates = chartDates.slice(-180);
        const displayPrices = chartPrices.slice(-180);

        // Send the clean, finalized data to the frontend as JSON
        res.json({
            cryptocurrency: cryptocurrency,
            predictedTrend: predictionResult.prediction,
            confidence: predictionResult.confidence,
            articles: articles,
            chartDates: displayDates,
            chartPrices: displayPrices
        });

    } catch (error) {
        console.error("Error fetching data:", error);
        res.status(500).send("Error fetching data");
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});