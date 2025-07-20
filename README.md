# Mining Student Digital Behavior Patterns (Association Rule Mining)

This project explores patterns in student digital behavior using **Association Rule Mining (ARM)**. It analyzes student activity logs to extract hidden associations that can help enhance learning outcomes and engagement.

## 👨‍💻 Authors
- **Kaye Marie Lagahid**
- **James Quijano**
- **Jhedver Tan**

## 🛠️ Tech Stack
- **Frontend**: Angular (TypeScript)
- **Backend**: Django + Django REST Framework (Python)
- **Mining Library**: `mlxtend` for Apriori / FP-Growth
- **Styling**: TailwindCSS (optional)

## 📁 Project Structure

```bash
pagmulat_app/
pagmulat_backend/
├── data/
│   ├── raw/
│   └── processed/
├── pagmulat_api/
│   ├── data_preprocessing_transformation/
│   │   ├── processors/
│   │   ├── mappings/
│   │   └── utils/
│   ├── feature_engineering/
│   ├── arm_mining/
│   ├── data_synthesis/
│   ├── model_training/
│   └── __init__.py
├── scripts/
└── requirements.txt
│
├── frontend/                  # Angular frontend
│   ├── angular.json
│   ├── package.json
│   └── src/
│       ├── app/
│       │   ├── components/
│       │   ├── services/
│       │   └── pages/
│       └── assets/
│
└── README.md
