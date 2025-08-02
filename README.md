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
├── pagmulat_frontend/
│   └── pagmulat/
│       ├── angular.json
│       ├── package.json
│       └── src/
│           ├── app/
│           │   ├── core/
│           │   │   ├── core.module.ts
│           │   │   └── services/
│           │   │       ├── api.service.ts
│           │   │       └── arm.service.ts
│           │   ├── shared/
│           │   │   ├── shared.module.ts
│           │   │   ├── pipes/
│           │   │   │   └── format-confidence.pipe.ts
│           │   │   └── components/
│           │   │       ├── file-uploader/
│           │   │       ├── metrics-card/
│           │   │       └── rules-table/
│           │   ├── pages/
│           │   │   ├── analysis/
│           │   │   │   ├── analysis.module.ts
│           │   │   │   ├── analysis.component.ts
│           │   │   │   ├── parameter-form/
│           │   │   │   │   └── parameter-form.component.ts
│           │   │   │   └── visualization/
│           │   │   │       └── visualization.component.ts
│           │   │   ├── dashboard/
│           │   │   │   ├── dashboard.module.ts
│           │   │   │   └── dashboard.component.ts
│           │   │   └── upload/
│           │   │       └── upload.module.ts
│           │   ├── services/
│           │   │   └── data.service.ts
│           │   ├── app.module.ts
│           │   ├── app-routing.module.ts
│           └── assets/
│               ├── images/
│               └── data/
│
└── README.md
