# 🌿 CropCare — Environmental Intelligence Platform for Farmers

CropCare is a Flask web app that helps farmers with:

| Feature | What it does | Powered by |
|---|---|---|
| **Crop recommendation** | Suggests the best crop from soil N/P/K, pH, rainfall and the live temperature/humidity of your city, plus growing tips | RandomForest model, **OpenWeatherMap**, **Groq** |
| **Fertilizer suggestion** | Compares your soil N/P/K with the ideal values for a crop and says what to fix | CSV reference table |
| **Disease detection** | Upload a leaf photo → disease name, cause and cure (38 classes) | ResNet9 PyTorch model |
| **Disease timeline tracking** | Track a plant over time: severity, affected area %, trend charts, alerts | **MongoDB Atlas** + OpenCV |
| **Market price insight** | Price range, best months to sell, top mandis, trend outlook | **Tavily** search + **Groq** |
| **Smart crop planning** | Full crop plan, cost-vs-revenue chart, activity calendar, PDF download | **Tavily** + **Groq** |
| **Hindi translation** | One-click Hindi version of AI-generated text | **Groq** |
| **Accounts** | Farmer registration/login with mobile number | Flask-Login |

Accounts, tracked plants and disease records are stored in **MongoDB Atlas** (cloud database, free tier available)
using the official **PyMongo** driver. You can browse the data in the Atlas website or in the **MongoDB Compass** desktop app.

---

## 1. Project structure

```
crop-care/
├── app/
│   ├── app.py                 # Flask routes and AI/ML logic
│   ├── config.py              # reads .env (MongoDB URI, API keys)
│   ├── database.py            # MongoDB (PyMongo) data layer: users, plants, records
│   ├── init_db.py             # tests the Atlas connection + creates indexes
│   ├── requirements.txt
│   ├── Data/fertilizer.csv    # ideal N/P/K per crop
│   ├── models/                # RandomForest.pkl, plant_disease_model.pth
│   ├── utils/                 # disease info, fertilizer text, severity analysis, ResNet9
│   ├── templates/             # HTML pages
│   └── static/                # css, js, images, uploads/ (leaf photos)
├── docker-compose.yml         # runs the web app in Docker (database = Atlas)
├── Dockerfile
├── render.yaml                # deploy on Render
├── .env.example               # copy to .env and fill in
└── README.md
```

---

## 2. What you need

* **Python 3.11 or newer** (the saved crop model needs scikit-learn 1.8, which requires Python 3.11+).
  Check with `python --version`. Or use Docker instead (it brings its own Python).
* A free **MongoDB Atlas** account (steps below).
* Free API keys:
  * OpenWeatherMap → https://openweathermap.org/api
  * Groq → https://console.groq.com/keys
  * Tavily → https://app.tavily.com
* About 3 GB free disk space (PyTorch is large).

---

## 3. Create your MongoDB Atlas database (≈ 5 minutes)

> Atlas screens are updated from time to time, so button names may differ slightly.

1. **Sign up** at https://www.mongodb.com/cloud/atlas/register and log in.
2. **Create a cluster**: click **Create** / **Build a Database** → choose the free **M0** tier →
   pick a provider and a region close to you (for India, e.g. AWS Mumbai `ap-south-1`) → **Create**.
3. **Create a database user** (left menu **Security → Database Access → Add New Database User**):
   * Authentication: *Password*. Choose a username and a password.
     Tip: use letters and numbers only for the password — it avoids URL-encoding problems.
   * Role: **Read and write to any database**. Click **Add User**.
4. **Allow your IP** (left menu **Security → Network Access → Add IP Address**):
   * Click **Add Current IP Address** (best), or for testing enter `0.0.0.0/0` (allows any IP).
   * Click **Confirm** and wait until the status says *Active*.
   * If you move to another network (home → office) you must add that IP too.
5. **Copy the connection string**: left menu **Database** → your cluster → **Connect** →
   **Drivers** (Python) → copy the string. It looks like:

   ```
   mongodb+srv://<db_username>:<db_password>@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority
   ```

   Replace `<db_username>` and `<db_password>` with the user you created (remove the `< >`).

   If your password contains special characters (`@ : / ? # %` …), URL-encode it:
   ```bash
   python -c "from urllib.parse import quote_plus; print(quote_plus('p@ss:word'))"
   ```

You do **not** need to create the database or collections by hand — CropCare creates the database
(`cropcare`) and its collections automatically on first use.

---

## 4. Configure the project

```bash
cp .env.example .env          # Windows PowerShell:  copy .env.example .env
```

Open `.env` in a text editor and set (no quotes around values):

| Variable | What to put |
|---|---|
| `SECRET_KEY` | A long random string. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `MONGODB_URI` | The Atlas connection string from step 5 above |
| `MONGODB_DB` | Database name, default `cropcare` |
| `WEATHER_API_KEY` | OpenWeatherMap key |
| `GROQ_API_KEY` | Groq key |
| `TAVILY_API_KEY` | Tavily key |

> 🔐 **Never commit `.env`, and never paste API keys or your MongoDB password in chats, screenshots or GitHub.**
> If a key or password is ever exposed, delete/rotate it in the provider's dashboard and create a new one.

---

## 5. Run the project

### Option A — Run with Python (recommended for development)

```bash
cd app

python -m venv venv
source venv/bin/activate           # Windows (PowerShell): venv\Scripts\Activate.ps1
                                   # Windows (cmd):        venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt    # downloads PyTorch - takes a few minutes

python init_db.py                  # ← test the Atlas connection first
python app.py                      # start the web app
```

`python init_db.py` should print `✅ Connected to MongoDB Atlas`. If it prints an error, fix it first
(see Troubleshooting) — the message tells you what is wrong.

Open **http://localhost:8000** in your browser.

### Option B — Run with Docker

```bash
docker compose up -d --build       # first build takes several minutes
docker compose logs -f web         # watch the logs (Ctrl+C to stop watching)
```

Open **http://localhost:8000**. Stop with `docker compose down`.
The container connects to Atlas using the `MONGODB_URI` in your `.env`, so your machine's IP must be allowed in Atlas Network Access.

### First use

1. Open the site → **Register** (name, 10-digit mobile number, password of 6+ characters).
2. Use the features from the home page. Health check: http://localhost:8000/health → `{"database":"up","status":"ok"}`.

---

## 6. Viewing your data

### In the Atlas website
**Database → your cluster → Browse Collections** → select database `cropcare` → pick a collection.
You can filter (e.g. `{ "mobile": "9876543210" }`), edit and delete documents there.

### In MongoDB Compass (desktop app)
1. Download Compass: https://www.mongodb.com/try/download/compass
2. Click **Add new connection**, paste the **same `MONGODB_URI`** from your `.env`, click **Connect**.
3. Open the `cropcare` database → `users`, `plants` or `disease_records`.

(Other desktop tools such as Studio 3T, NoSQLBooster, or the VS Code "MongoDB for VS Code" extension work with the same URI.)

### Collections and fields

| Collection | Fields |
|---|---|
| `users` | `_id`, `name`, `mobile` (unique), `password_hash`, `created_at` |
| `plants` | `_id`, `user_id`, `plant_id` (short code, unique), `plant_name`, `crop_type`, `location`, `created_at` |
| `disease_records` | `_id`, `plant_ref` (→ plants._id), `user_id`, `image_filename`, `disease_class`, `confidence`, `severity_score`, `affected_area_pct`, `timestamp` (observation date), `notes`, `created_at` |

Passwords are never stored in plain text — only a salted hash.

Useful Compass/Atlas filters on `disease_records`:
```js
{ "disease_class": /Late_blight/ }          // all late blight detections
{ "severity_score": { "$gte": 60 } }        // severe cases
```

---

## 7. Useful commands

```bash
cd app
python init_db.py                 # test connection + create indexes (safe, keeps data)
python init_db.py --reset         # DELETE all CropCare data in Atlas (asks first)
```

Indexes created automatically: unique `users.mobile`, unique `plants.plant_id`, and lookup indexes for plants/records.

---

## 8. Troubleshooting

| Problem / message | Fix |
|---|---|
| `MONGODB_URI is not set` | You have no `.env`, or it is in the wrong folder. Put `.env` in the project root (next to `README.md`) or in `app/`. |
| `Could not reach MongoDB Atlas` (timeout) | Your IP is not allowed → Atlas **Network Access → Add IP Address**. Also check internet/VPN/firewall and that the cluster is not paused. |
| `Atlas rejected the username/password` | Check the user under **Database Access**; re-type the password in `MONGODB_URI`; URL-encode special characters. |
| `not a valid MongoDB connection string` | Re-copy the string from Atlas; make sure you removed the `< >` brackets and there are no quotes or spaces. |
| `Could not resolve the Atlas cluster address` / DNS error | Check the cluster host name in the URI and your internet. Run `pip install -r requirements.txt` again (needs `dnspython`). |
| `SSL: CERTIFICATE_VERIFY_FAILED` | `pip install --upgrade certifi`. On macOS also run the "Install Certificates.command" that comes with Python. Corporate proxies may block Atlas. |
| Crop recommendation shows "try again" | Check `WEATHER_API_KEY` (new keys can take up to a couple of hours to activate) and the city. |
| Tips / translation / market / planning fail | Check `GROQ_API_KEY` and `TAVILY_API_KEY`; look at the terminal log for the exact error. A retired Groq model can be changed via `GROQ_*_MODEL` in `.env`. |
| `pip install` can't find `torch==2.2.0+cpu` | Use Python 3.11 and keep the `--extra-index-url` line in `requirements.txt`. |
| Model fails to load / `InconsistentVersionWarning` | Install exactly `scikit-learn==1.8.0` (Python 3.11+). |
| Registration says "Database is temporarily unavailable" | Atlas became unreachable (IP changed, cluster paused). Check `/health` and Atlas Network Access. |

---

## 9. What was fixed in this version

**Database (SQLAlchemy/SQLite → MongoDB Atlas)**
* Replaced SQLAlchemy with **PyMongo**; connection string comes from `MONGODB_URI` in `.env`.
* Clear start-up error messages for the usual Atlas problems (IP not allowed, wrong password, bad URI, DNS).
* Unique indexes on mobile number and plant code; duplicate registrations are rejected by the database itself.
* Ownership is checked on every plant/record read and delete; a failed save no longer leaves an empty plant behind.
* Removed the unused `environment_data` and `treatments` tables (nothing in the app used them).
* `init_db.py` no longer wipes data; it only tests the connection and creates indexes (`--reset` is explicit and asks first).

**Crashes / wrong results**
* Detecting **Tomato Target Spot** crashed (typo in the disease table key `Tomato___Target_Spo`).
* Disease detection crashed on **PNG with transparency, grayscale, or non-square photos** → images are converted to RGB and resized to the 256×256 the model expects.
* **Fertilizer advice** picked the wrong nutrient on ties and crashed on unknown crops/bad input.
* **Crop planning** failed when the budget contained commas/symbols or when the AI wrote `₹50,000`; the crop name in the chart title was often a random word.
* Search queries had a hard-coded year; now the current year is used.
* Crop prediction passes named columns (as the model was trained) and accepts decimal N/P/K values.

**Security**
* Open-redirect in the login `next` parameter fixed.
* `/translate-hindi` required no login (anyone could spend your Groq quota) → now login-protected and validated.
* PDF export no longer trusts raw HTML from the browser and no longer leaks error text.
* Uploaded images are verified as real images and stored under random names; orphan files are cleaned up on errors.
* Session cookies are `HttpOnly` + `SameSite=Lax`; start-up warns if `SECRET_KEY` is missing; secrets come only from `.env`; `.gitignore` added.

**Reliability**
* Charts use Matplotlib's thread-safe `Figure` API (the old `pyplot` global state is unsafe with gunicorn threads) and no longer merge records taken on the same day.
* Relative paths for models/data/uploads are now absolute, so the app runs from any directory.
* Error messages are shown on every page (flash messages added to the layout); failures of Groq/Tavily/OpenWeatherMap are logged instead of silently swallowed.
* `/health` endpoint (checks the database) used by the Docker health check.

---

## 10. Deploying on Render

`render.yaml` defines the web service. In the Render dashboard add the environment variables
`MONGODB_URI`, `WEATHER_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`.
In Atlas **Network Access**, allow `0.0.0.0/0`, because Render's outgoing IP addresses are not fixed
(keep a strong database password).
Uploaded photos live on the service disk (ephemeral on the free plan); add a persistent disk at `/app/static/uploads` to keep them.

---

## 11. Notes and limitations

* The disease model recognises the 38 PlantVillage leaf classes (apple, tomato, potato, grape, corn, …) and works best on a single clear leaf photo.
* "Affected area %" is a colour-based estimate and "severity" is a heuristic (disease type × model confidence) — guidance, not a diagnosis.
* AI-generated advice (tips, prices, plans) can be wrong; verify prices with your local mandi before selling.
* Uploaded photos are stored on the server's disk (not in MongoDB); the database stores only the file name.
* Forms rely on `SameSite` cookies rather than CSRF tokens. If you expose the app publicly, consider adding Flask-WTF CSRF protection.
* Don't run gunicorn with `--preload` (a MongoDB client must be created after the worker process forks). The provided Dockerfile is already correct.

## License

See `LICENSE`.
