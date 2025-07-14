<div align="center">

# ⚡ RE.AI - Returns Management Platform

A sophisticated Django-based returns management platform that leverages advanced analytics to optimize product return decisions through demand forecasting, profitability analysis, and location optimization.

> **📊 Built for Walmart Sparkathon 2025**

---

</div>

## ✨ Core Features

### 🎯 Intelligent Decision Engine

- **Smart Classification**
  - `⚡ Automatic` categorization of returns into Restock, Refurbish, or Recycle
  - `🔄 Real-time` analysis of product condition and market demand
  - `📊 Smart` routing based on confidence scores

- **Market Analysis**
  - `📈 Dynamic` store-specific demand forecasting
  - `💡 Intelligent` profit potential calculations
  - `📍 Strategic` restock location recommendations

### 🔄 Streamlined Workflow

<details>
<summary><b>Dashboard Overview</b></summary>

- Comprehensive statistics and performance metrics
- Real-time processing status
- Quick-access action items
</details>

<details>
<summary><b>Process Management</b></summary>

- Automated review system for high-confidence decisions
- Manual review interface for complex cases
- Digital invoice generation with detailed breakdowns
</details>

### 🎨 Modern Interface

- `⚡ Clean`, intuitive dashboard design
- `🔄 Real-time` updates and predictions
- `📱 Responsive` layouts for all devices
- `📊 Interactive` data visualizations

<br>

## 🛠 Technical Architecture

### Core Stack
```
📦 Backend    │ Django + PostgreSQL
🎨 Frontend   │ Django Templates + Tailwind CSS
🧠 Analytics  │ Gradient Boosting, Random Forest
🗄️ Storage    │ Supabase
```

### Infrastructure
| Service | Technology |
|---------|------------|
| 🌐 Hosting | Render (MVP) |
| 🗃️ Database | Supabase |
| 📂 Storage | Supabase Object Storage |

<br>

## 🚀 Development Setup

<details>
<summary><b>🔧 Environment Setup</b></summary>

```bash
# Clone repository
git clone <repo-url>
cd RE.AI

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><b>⚙️ Configuration</b></summary>

Create `.env` file with required credentials:
```env
DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=your-database-url

# Supabase Configuration
supabase_url = 'https://your-project.supabase.co'
supabase_service_key = 'your-secret-key'
```
</details>

<details>
<summary><b>▶️ Launch Application</b></summary>

```bash
python manage.py migrate
python manage.py runserver
```
</details>

<br>

## 📁 Project Structure
```
RE.AI/
├── reai/               # Core Configuration
│   └── settings.py, urls.py
│
├── returns/            # Main Application
│   ├── models/        # ML Models
│   ├── templates/     # Interface Views
│   ├── static/       # Assets
│   └── views.py      # Business Logic
│
├── management/        # Custom Commands
└── requirements.txt
```

<br>

## 🧠 Analytics Engine

### Model Architecture
| Component | Implementation |
|-----------|---------------|
| 📊 Demand Forecasting | Gradient Boosting (25 features) |
| 💰 Profit Prediction | Random Forest (9 features) |
| 📚 Dataset | 500,000+ transactions across 30 stores |
| ⚡ Performance | Sub-2-second inference time |

### Processing Flow
1. 📝 Return submission with product details
2. 🔄 Classification and routing decision
3. For restocking candidates:
   - 📊 Multi-location demand analysis
   - 💰 Profit potential calculations
   - ✨ Confidence scoring
4. 🔄 Automated or manual review based on thresholds
5. 📄 Digital invoice generation upon approval

<br>

## 🌐 Deployment

### Live Platform
<div align="center">

**[🔗 https://re-ai.onrender.com](https://re-ai.onrender.com)**

**[🎥 View Demo](https://www.youtube.com/watch?v=n7L1jRpCcO8)**

</div>

<br>

## 📋 Evaluation Notes

### Testing Guidelines
| Feature | Description |
|---------|-------------|
| 🧪 Demo | Functional demo with test dataset |
| 📊 Workflow | Clear process visualization |
| ⚡ Automation | Automated and manual flows |
| 💰 ROI | Cost-effective decision making |

### Roadmap
- 📄 PDF invoice generation
- 🔔 Alert system for critical cases
- 🔄 ERP system integration
- 🌐 Vendor API ecosystem
- 📊 Enhanced analytics visualization

<br>

<div align="center">

---

### *Optimizing Returns Management Through Data-Driven Decisions* ###

</div>