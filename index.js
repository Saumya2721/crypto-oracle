import express from "express";
import axios from "axios";
import MultivariateLinearRegression from "ml-regression-multivariate-linear";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;


app.set("view engine", "ejs");
app.use(express.static("public"));
app.use(express.urlencoded({extended: true}));

app.get("/", (req,res) => {
    res.render("index.ejs");
});

app.post("/analyze", async (req,res) => {
    const cryptocurrency = req.body.cryptocurrency;
    const market = "USD";

    try {
        console.log(`Fetching data for ${cryptocurrency}...`);
        // Fetch Market Data (Alpha Vantage)
        // Constructing the URL
        const alphaVantageURL = `https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=${cryptocurrency}&market=${market}&apikey=${process.env.ALPHA_VANTAGE_KEY}`;
        
        // Await axios.get()
        // Extract the last 30 days of  prices from the JSON response
        const response = await axios.get(alphaVantageURL);
        console.log("ALPHA VANTAGE RESPONSE:", response.data);
        const timeSeries = response.data["Time Series (Digital Currency Daily)"];

        // Throw an error if timeSeries is undefined
        if (!timeSeries) {
            throw new Error("Invalid API response: Time Series data not found");
        }

        // 1. Get ALL keys
        const allDates = Object.keys(timeSeries);

        // 2. Sort by date string (Newest first)
        allDates.sort((a, b) => new Date(b) - new Date(a));

        // 3. Take the 30 most recent, then reverse to get [Oldest -> Newest] for the chart
        const dates = allDates.slice(0, 30).reverse();

       // Use the exact keys from your terminal output!
        const openPrices = dates.map(date => parseFloat(timeSeries[date]['1. open']));
        const closePrices = dates.map(date => parseFloat(timeSeries[date]['4. close']));
        const volumes = dates.map(date => parseFloat(timeSeries[date]['5. volume']));
        const DailyHighs = dates.map(date => parseFloat(timeSeries[date]['2. high']));
        const DailyLows = dates.map(date => parseFloat(timeSeries[date]['3. low']));

       // Fetch News Data (The Guardian)
        // Add "%20crypto" to the query to force context
        // 1. Force the word 'crypto' to be required
        // 2. Restrict the search to only the Business or Technology sections
        const guardianURL = `https://content.guardianapis.com/search?q=${cryptocurrency}%20AND%20crypto&section=business|technology&order-by=newest&api-key=${process.env.NEWS_API_KEY}`;

        const newsResponse = await axios.get(guardianURL);
        
        // Add a fallback in case results are still empty for an obscure coin
        const rawArticles = newsResponse.data.response.results || [];
        
        const articles = rawArticles.slice(0, 5).map(article => ({
            title: article.webTitle,
            url: article.webUrl
        }));

        // We slice to match the 30 days of data we actually analyzed
        const chartDates = dates.slice(1); 
        const chartPrices = closePrices.slice(1);

        // ML MODEL : Predicting Tomorrow's Price

        // 5. DEFINE 'Y' HERE (Crucial Fix!)
        // Wraps each closing price in an array for the ML model
        const y = closePrices.slice(1).map(price => [price]);

        // ==========================================
        // ACCURACY TESTING
        // ==========================================
        function getMAE(actualY, predictedY) {
            let errorSum = 0;
            for (let i = 0; i < actualY.length; i++) {
                errorSum += Math.abs(actualY[i][0] - predictedY[i][0]);
            }
            return (errorSum / actualY.length).toFixed(2);
        }

        // Model 1: ALL 4 FEATURES
        const X_all = openPrices.slice(0, -1).map((price, index) => [price, volumes[index], DailyHighs[index], DailyLows[index]]);
        const modelAll = new MultivariateLinearRegression(X_all, y);
        const predictionsAll = modelAll.predict(X_all);
        const allError = getMAE(y, predictionsAll);
        console.log(`Model (All 4 Features) Mean Error: $${allError}`);

        // Model 2: JUST OPEN & VOLUME
        const X_basic = openPrices.slice(0, -1).map((price, index) => [price, volumes[index]]);
        const modelBasic = new MultivariateLinearRegression(X_basic, y);
        const predictionsBasic = modelBasic.predict(X_basic);
        const basicError = getMAE(y, predictionsBasic);
        console.log(`Model (Open & Vol) Mean Error: $${basicError}`);

        // ==========================================
        // PREDICT TOMORROW'S PRICE
        // ==========================================
        // Get today's actual data to feed into the model for tomorrow's prediction
        const todayOpen = openPrices[openPrices.length - 1];
        const todayVolume = volumes[volumes.length - 1];
        const todayHigh = DailyHighs[DailyHighs.length - 1];
        const todayLow = DailyLows[DailyLows.length - 1];
        
        // Predict! (Returns a 2D array, so we grab [0][0])
        const tomorrowPrediction = modelAll.predict([[todayOpen, todayVolume, todayHigh, todayLow]])[0][0];

        // Send the clean, finalized data to the frontend
        res.render("index.ejs", {
            cryptocurrency: cryptocurrency,
            predictedPrice: tomorrowPrediction.toFixed(2),
            errorMargin: allError,
            articles: articles,
            chartDates: chartDates,
            chartPrices: chartPrices
        });

    } catch (error) {
        console.error("Error fetching data:", error);
        res.status(500).send("Error fetching data");
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});