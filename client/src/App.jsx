import { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import axios from 'axios';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend
);

function App() {
  const [cryptocurrency, setCryptocurrency] = useState('');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [chartRange, setChartRange] = useState(180);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setData(null);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3000';
      const response = await axios.post(`${apiUrl}/analyze`, {
        cryptocurrency
      });
      setData(response.data);
    } catch (err) {
      setError('Error fetching data. Please ensure the backend is running and the ticker is valid.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const displayDates = data ? data.chartDates.slice(-chartRange) : [];
  const displayPrices = data ? data.chartPrices.slice(-chartRange) : [];

  const chartData = data ? {
    labels: displayDates,
    datasets: [
      {
        label: 'Closing Price (USD)',
        data: displayPrices,
        borderColor: '#0d6efd',
        backgroundColor: 'rgba(13, 110, 253, 0.1)',
        borderWidth: 2,
        pointRadius: 3,
        fill: true,
        tension: 0.3,
      },
    ],
  } : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: false }
    },
    scales: {
      y: {
        ticks: {
          callback: function (value) { return '$' + value; }
        }
      }
    }
  };

  return (
    <div className="bg-light min-vh-100">
      <nav className="navbar navbar-dark bg-dark mb-4 shadow-sm">
        <div className="container">
          <a className="navbar-brand" href="/">Crypto Oracle</a>
        </div>
      </nav>

      <div className="container mt-5">
        <div className="card p-4 mx-auto mb-5 shadow-sm border-0" style={{ maxWidth: '500px' }}>
          <form onSubmit={handleSubmit}>
            <div className="mb-3">
              <label htmlFor="cryptoInput" className="form-label fw-bold">Search an Asset:</label>
              <input
                className="form-control"
                list="cryptoOptions"
                name="cryptocurrency"
                id="cryptoInput"
                placeholder="Type a ticker (e.g., BTC)..."
                required
                autoComplete="off"
                value={cryptocurrency}
                onChange={(e) => setCryptocurrency(e.target.value)}
              />
              <datalist id="cryptoOptions">
                <option value="BTC">Bitcoin</option>
                <option value="ETH">Ethereum</option>
                <option value="SOL">Solana</option>
                <option value="XRP">Ripple</option>
                <option value="DOGE">Dogecoin</option>
                <option value="ADA">Cardano</option>
                <option value="LINK">Chainlink</option>
                <option value="DOT">Polkadot</option>
              </datalist>
            </div>
            <button type="submit" className="btn btn-primary w-100 fw-bold" disabled={loading}>
              {loading ? 'Consulting...' : 'Consult the Oracle'}
            </button>
          </form>
          {error && <div className="alert alert-danger mt-3">{error}</div>}
        </div>

        {data && (
          <>
            <div className="row mb-4">
              <div className="col-12">
                <h3 className="border-bottom pb-2">Analysis for {data.cryptocurrency}</h3>
              </div>
            </div>

            <div className="row text-center mb-4">
              <div className="col-md-6 mb-3">
                <div className="card p-4 bg-white shadow-sm border-0 h-100">
                  <h5 className="text-muted text-uppercase" style={{ fontSize: '0.9rem' }}>Tomorrow's Trend Prediction</h5>
                  <h1 className={`display-5 fw-bold ${data.predictedTrend === 'BULLISH' ? 'text-success' : data.predictedTrend === 'BEARISH' ? 'text-danger' : 'text-secondary'}`}>{data.predictedTrend}</h1>
                </div>
              </div>
              <div className="col-md-6 mb-3">
                <div className="card p-4 bg-white shadow-sm border-0 h-100">
                  <h5 className="text-muted text-uppercase" style={{ fontSize: '0.9rem' }}>Model Confidence</h5>
                  <h1 className="text-primary display-5 fw-bold">{data.confidence}%</h1>
                  <small className="text-muted">XGBoost Classifier (~5 Years Data)</small>
                </div>
              </div>
            </div>

            <div className="row mb-5">
              <div className="col-12">
                <div className="card p-4 bg-white shadow-sm border-0">
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <h4 className="mb-0">Price History</h4>
                    <div className="btn-group btn-group-sm" role="group">
                      <button type="button" className={`btn ${chartRange === 7 ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setChartRange(7)}>1W</button>
                      <button type="button" className={`btn ${chartRange === 30 ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setChartRange(30)}>1M</button>
                      <button type="button" className={`btn ${chartRange === 90 ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setChartRange(90)}>3M</button>
                      <button type="button" className={`btn ${chartRange === 180 ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setChartRange(180)}>6M</button>
                    </div>
                  </div>
                  <Line data={chartData} options={chartOptions} />
                </div>
              </div>
            </div>

            <div className="row">
              <div className="col-12">
                <h4 className="mb-3">Recent Headlines</h4>
                <div className="list-group shadow-sm mb-5">
                  {data.articles && data.articles.map((article, index) => (
                    <a key={index} href={article.url} target="_blank" rel="noopener noreferrer" className="list-group-item list-group-item-action py-3 border-0 border-bottom">
                      <h5 className="mb-1">{article.title}</h5>
                      <p className="mb-1 text-muted small">{article.description}</p>
                    </a>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      <footer className="text-center mt-5 pb-4 text-muted">
        <small>&copy; {new Date().getFullYear()} | Built with Binance & NEWSDATA.IO APIs</small>
      </footer>
    </div>
  );
}

export default App;
