# Mining Student Digital Behavior Patterns (Association Rule Mining)

This project explores patterns in student digital behavior using **Association Rule Mining (ARM)**. It analyzes student activity logs to extract hidden associations that can help enhance learning outcomes and engagement.

## 👨‍💻 Authors
- **Lagahid**
- **Quijano**
- **Tan**

## 🛠️ Tech Stack
- **Frontend**: Angular (TypeScript)
- **Backend**: Django + Django REST Framework (Python)
- **Mining Library**: `mlxtend` for Apriori / FP-Growth
- **Styling**: TailwindCSS (optional)

## 📁 Project Structure

```bash
pagmulat_app/
├── backend/                   # Django backend
│   ├── manage.py
│   ├── requirements.txt
│   ├── pagmulat/              # Django project files
│   └── mining/                # App for data mining logic
│       ├── models.py
│       ├── views.py
│       ├── urls.py
│       └── arm/               # Association Rule Mining logic
│           ├── apriori.py
│           └── preprocessing.py
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
