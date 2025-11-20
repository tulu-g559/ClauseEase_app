import os
import re
from collections import Counter, defaultdict
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, send_file, session
from werkzeug.utils import secure_filename

from mod1_docingestion import extract_text
from mod2_preprocess import preprocess_contract_text
from mod3_legalClause import detect_clause_type
from mod4_legalTermRec import recognize_legal_terms, legal_terms
from mod5_LangSimple import simplify_text


try:
    from viz.plot_clause_distribution import plot_clause_distribution
    from viz.plot_term_frequency import plot_top_terms
    from viz.plot_readability_impact import plot_readability
except Exception:
    plot_clause_distribution = None
    plot_top_terms = None
    plot_readability = None
import json
import io

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'dev-secret-for-demo'  # Required for session management

os.makedirs(UPLOAD_FOLDER, exist_ok=True)



##=====================
#   Databse Integration
##===================== 

import firebase_admin
from firebase_admin import credentials, firestore
import hashlib

# Initialize Firebase Admin SDK
cred = credentials.Certificate("data/clauseease-service-key.json")  # Path to your key file
firebase_admin.initialize_app(cred)
db = firestore.client()
#======================


# Login required decorator
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please log in to access this page.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please log in to access this page.')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------
# Visualization helper methods
# -----------------------------
VOWELS = "aeiouy"


def syllable_count(word: str) -> int:
    w = re.sub(r'[^a-z]', '', (word or '').lower())
    if not w:
        return 0
    groups = re.findall(r'[aeiouy]+', w)
    count = len(groups)
    if w.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def flesch_score(text: str) -> float:
    if not text:
        return 0.0
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\w+', text)
    if not sentences or not words:
        return 0.0
    syllables = sum(syllable_count(w) for w in words)
    try:
        score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
        return round(score, 1)
    except ZeroDivisionError:
        return 0.0


def normalize_clause_type(value):
    if isinstance(value, (list, tuple)):
        try:
            return ', '.join([str(v) for v in value if v])
        except Exception:
            return str(value)
    return value or 'Unknown'


def strip_clause_confidence(results):
    if not results:
        return results
    for r in results:
        t = r.get('type')
        label = t
        confidence = r.get('confidence')
        if isinstance(t, (list, tuple)):
            label = t[0] if t else 'Unknown'
            if len(t) > 1:
                confidence = float(t[1])
        r['type'] = label
        if confidence is not None:
            try:
                r['confidence'] = float(confidence)
            except (TypeError, ValueError):
                r['confidence'] = 0.0
    return results


def build_chart_data(results: list) -> dict:
    if not results:
        return {}

    clause_counts = Counter()
    term_counts = Counter()
    per_type_scores = defaultdict(list)
    overall_before = []
    overall_after = []

    for r in results:
        ctype = normalize_clause_type(r.get('type'))
        clause_counts[ctype] += 1
        terms = r.get('terms') or {}
        if isinstance(terms, dict):
            for t in terms.keys():
                term_counts[t] += 1
        elif isinstance(terms, (list, tuple)):
            for t in terms:
                term_counts[t] += 1
        before = flesch_score(r.get('cleaned') or r.get('raw'))
        after = flesch_score(r.get('simple') or r.get('simplified'))
        per_type_scores[ctype].append((before, after))
        overall_before.append(before)
        overall_after.append(after)

    clause_labels = list(clause_counts.keys())
    clause_values = [clause_counts[label] for label in clause_labels]

    top_terms = term_counts.most_common(10)
    term_labels = [label for label, _ in top_terms]
    term_values = [value for _, value in top_terms]

    readability_labels = ['Overall']
    readability_before = [round(sum(overall_before) / len(overall_before), 1)] if overall_before else [0]
    readability_after = [round(sum(overall_after) / len(overall_after), 1)] if overall_after else [0]

    top_types = [label for label, _ in clause_counts.most_common(5)]
    for label in top_types:
        vals = per_type_scores.get(label, [])
        if not vals:
            continue
        before_avg = sum(b for b, _ in vals) / len(vals)
        after_avg = sum(a for _, a in vals) / len(vals)
        readability_labels.append(label)
        readability_before.append(round(before_avg, 1))
        readability_after.append(round(after_avg, 1))

    return {
        'clauseDistribution': {
            'labels': clause_labels,
            'values': clause_values
        },
        'termFrequency': {
            'labels': term_labels,
            'values': term_values
        },
        'readability': {
            'labels': readability_labels,
            'before': readability_before,
            'after': readability_after
        }
    }





##==================================
#  ROUTES SECTION1: Signup and Login
##==================================

@app.route('/')
def index():
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('login.html')

    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    # role may be 'user' or 'admin'
    role = request.form.get('role', 'user')

    if not email or not password or not name:
        flash('Please fill all fields.', 'error')
        return redirect(url_for('index'))

    # Check if user already exists in Firestore
    user_ref = db.collection('users').document(email)
    if user_ref.get().exists:
        flash('Account already exists. Please log in.', 'error')
        return redirect(url_for('index'))

    # Hash password before storing (for security)
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    # Store user in Firestore (including role)
    user_ref.set({
        'name': name,
        'email': email,
        'password_hash': password_hash,
        'role': role,
        'created_at': firestore.SERVER_TIMESTAMP
    })

    # Start session
    session['user_email'] = email
    session['user_name'] = name
    session['role'] = role
    flash('Registration successful! Welcome.', 'success')
    return redirect(url_for('dashboard'))


#-----------------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash('Please enter both email and password.', 'error')
            return redirect(url_for('login'))

        user_ref = db.collection('users').document(email)
        user_doc = user_ref.get()

        if not user_doc.exists:
            flash('No account found with that email.', 'error')
            return redirect(url_for('login'))

        user_data = user_doc.to_dict()
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        if user_data.get('password_hash') == password_hash:
            session['user_email'] = email
            session['user_name'] = user_data.get('name', '')
            # populate role from stored user record
            session['role'] = user_data.get('role', 'user')
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect password.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))








##============================
#  ROUTES SECTION 2: Dashboard 
##============================

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    uploaded_file_info = None
    results = None
    logo_filename = None
    history = []
    simp_level = None
    # generated chart filenames (basenames in uploads/)
    clause_chart = None
    term_chart = None
    readability_chart = None
    # Analysis summary counts
    clause_count = 0
    term_count = 0
    chart_data = {}
    candidate = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
    if os.path.exists(candidate):
        logo_filename = 'logo.png'
    if request.method == 'POST':
        # handle uploaded document
        # if a logo file was uploaded via a field named 'logo', save it as logo.png
        if 'logo' in request.files and request.files['logo'].filename != '':
            logo_file = request.files['logo']
            if logo_file and allowed_file(logo_file.filename):
                logo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
                logo_file.save(logo_path)
                logo_filename = 'logo.png'

        if 'document' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['document']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            uploaded_file_info = {'name': filename, 'path': save_path}

            # Run pipeline: extract -> preprocess -> detect -> terms -> simplify
            contract_text = extract_text(save_path)
            if contract_text.startswith('[ERROR]'):
                flash(contract_text)
                return redirect(request.url)

            processed_clauses = preprocess_contract_text(contract_text)
            clause_types = [detect_clause_type(c['cleaned_text']) for c in processed_clauses]
            clause_terms = [recognize_legal_terms(c['cleaned_text'], legal_terms) for c in processed_clauses]
            # read simplification level chosen by user (Basic / Intermediate / Advanced)
            simp_level = request.form.get('simp_level', 'Intermediate')
            simplified = [simplify_text(c['cleaned_text'], level=simp_level) for c in processed_clauses]

            results = []
            for i, c in enumerate(processed_clauses):
                ct = clause_types[i]
                if isinstance(ct, (list, tuple)):
                    label = ct[0]
                    confidence = float(ct[1]) if len(ct) > 1 else 0.0
                else:
                    label = ct
                    confidence = 0.0
                results.append({
                    'index': i+1,
                    'raw': c['raw_text'],
                    'cleaned': c['cleaned_text'],
                    'type': label or 'Unknown',
                    'confidence': confidence,
                    'terms': clause_terms[i],
                    'simple': simplified[i]
                })
            results = strip_clause_confidence(results)
            # compute unique counts (unique cleaned clauses, unique term names)
            try:
                unique_clauses = {r.get('cleaned') for r in results if r.get('cleaned')}
                unique_terms = set()
                for r in results:
                    terms = r.get('terms') or {}
                    if isinstance(terms, dict):
                        for t in terms.keys():
                            unique_terms.add(t)
                    elif isinstance(terms, (list, tuple)):
                        for t in terms:
                            unique_terms.add(t)
                clause_count = len(unique_clauses)
                term_count = len(unique_terms)
            except Exception:
                clause_count = 0
                term_count = 0
            # persist results to a JSON file next to the uploaded file so we can
            # generate PDFs later or serve them for download
            try:
                json_path = save_path + '.results.json'
                # store results along with simplification level for future downloads/metadata
                out_payload = {
                    'results': results,
                    'simplification_level': simp_level
                }
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(out_payload, jf, ensure_ascii=False, indent=2)
                    # generate server-side visualizations (PNG) next to results json
                    try:
                        base = os.path.splitext(os.path.basename(json_path))[0]
                        clause_chart = None
                        term_chart = None
                        readability_chart = None
                        clause_img = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_clause_distribution.png")
                        term_img = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_top_terms.png")
                        read_img = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_readability_impact.png")
                        if plot_clause_distribution:
                            plot_clause_distribution(results, clause_img)
                        if plot_top_terms:
                            # top 10 terms
                            plot_top_terms(results, top_n=10, out_path=term_img)
                        if plot_readability:
                            plot_readability(results, read_img)
                        # expose basenames to template only if files exist
                        if os.path.exists(clause_img):
                            clause_chart = os.path.basename(clause_img)
                        if os.path.exists(term_img):
                            term_chart = os.path.basename(term_img)
                        if os.path.exists(read_img):
                            readability_chart = os.path.basename(read_img)
                    except Exception as e:
                        app.logger.warning(f"Could not generate visualization images: {e}")
                # Also save a metadata record in Firestore under users/<email>/analyses
                try:
                    import datetime
                    user_email = session.get('user_email')
                    if user_email:
                        analysis_doc = {
                            'filename': filename,
                            'uploaded_at': datetime.datetime.utcnow(),
                            'results_path': json_path,
                            'user_email': user_email,
                            'simplification_level': simp_level
                        }
                        user_analyses_ref = db.collection('users').document(user_email).collection('analyses')
                        new_doc = user_analyses_ref.document()
                        new_doc.set(analysis_doc)
                        # store the created analysis id so caller could redirect or show
                        analysis_id = new_doc.id
                except Exception as e:
                    app.logger.warning(f"Could not save analysis metadata to Firestore: {e}")
            except Exception as e:
                app.logger.warning(f"Could not save results JSON: {e}")
            chart_data = build_chart_data(results)
    # After POST or on GET, load user's history from Firestore
    try:
        user_email = session.get('user_email')
        if user_email:
            user_analyses_ref = db.collection('users').document(user_email).collection('analyses')
            docs = user_analyses_ref.order_by('uploaded_at', direction=firestore.Query.DESCENDING).stream()
            history = []
            for d in docs:
                data = d.to_dict()
                history.append({
                    'id': d.id,
                    'filename': data.get('filename'),
                    'uploaded_at': data.get('uploaded_at')
                })
    except Exception as e:
        app.logger.warning(f"Could not load user history: {e}")

    # Load analysis results either from current upload or from history
    load_analysis_id = request.args.get('analysis_id')
    # Try to load analysis from history if no current results and analysis_id is provided
    if not results and load_analysis_id and session.get('user_email'):
        try:
            doc_ref = db.collection('users').document(session.get('user_email')).collection('analyses').document(load_analysis_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                rp = data.get('results_path')
                if rp and os.path.exists(rp):
                    with open(rp, 'r', encoding='utf-8') as jf:
                        loaded = json.load(jf)
                        # support old format (list) and new format {results: [], simplification_level: ''}
                        if isinstance(loaded, dict) and 'results' in loaded:
                            results = loaded.get('results')
                            simp_level = loaded.get('simplification_level') or data.get('simplification_level')
                        else:
                            results = loaded
                            simp_level = data.get('simplification_level')
                        results = strip_clause_confidence(results)
                        uploaded_file_info = {'name': data.get('filename')}
                        # compute counts for loaded analysis
                        try:
                            unique_clauses = {r.get('cleaned') for r in results if r.get('cleaned')}
                            unique_terms = set()
                            for r in results:
                                terms = r.get('terms') or {}
                                if isinstance(terms, dict):
                                    for t in terms.keys():
                                        unique_terms.add(t)
                                elif isinstance(terms, (list, tuple)):
                                    for t in terms:
                                        unique_terms.add(t)
                            clause_count = len(unique_clauses)
                            term_count = len(unique_terms)
                        except Exception:
                            clause_count = 0
                            term_count = 0
                        chart_data = build_chart_data(results)
                    # check if visualization PNGs exist for this results file
                    base = os.path.splitext(os.path.basename(rp))[0]
                    candidate_clause = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_clause_distribution.png")
                    candidate_term = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_top_terms.png")
                    candidate_read = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_readability_impact.png")
                    if os.path.exists(candidate_clause):
                        clause_chart = os.path.basename(candidate_clause)
                    if os.path.exists(candidate_term):
                        term_chart = os.path.basename(candidate_term)
                    if os.path.exists(candidate_read):
                        readability_chart = os.path.basename(candidate_read)
        except Exception as e:
            app.logger.warning(f"Could not load analysis {load_analysis_id}: {e}")

    # Determine which tab to show
    active_tab = 'upload' if results else 'home'

    return render_template('dashboard.html', 
                       uploaded=uploaded_file_info,
                       results=results,
                       logo_filename=logo_filename,
                       history=history,
                       active_tab=active_tab,
                       clause_count=clause_count,
                       term_count=term_count,
                       simp_level=simp_level,
                       clause_chart=clause_chart,
                       term_chart=term_chart,
                      readability_chart=readability_chart,
                      chart_data=chart_data or {})


@app.route('/regenerate_charts', methods=['POST'])
@login_required
def regenerate_charts():
    # Expecting a filename (uploaded filename) to find the results json
    filename = request.form.get('filename')
    if not filename:
        flash('No filename provided to regenerate charts.', 'error')
        return redirect(url_for('dashboard'))

    json_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + '.results.json')
    if not os.path.exists(json_path):
        flash('Results JSON not found for provided filename.', 'error')
        return redirect(url_for('dashboard'))

    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            loaded = json.load(jf)
        if isinstance(loaded, dict) and 'results' in loaded:
            results = loaded.get('results')
        else:
            results = loaded
        results = strip_clause_confidence(results)

        base = os.path.splitext(os.path.basename(json_path))[0]
        clause_img = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_clause_distribution.png")
        term_img = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_top_terms.png")
        read_img = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_readability_impact.png")

        if plot_clause_distribution:
            plot_clause_distribution(results, clause_img)
        if plot_top_terms:
            plot_top_terms(results, top_n=10, out_path=term_img)
        if plot_readability:
            plot_readability(results, read_img)

        flash('Charts regenerated successfully.', 'success')
    except Exception as e:
        app.logger.warning(f"Could not regenerate charts: {e}")
        flash('Could not regenerate charts. Check server logs.', 'error')

    # Redirect back to dashboard showing upload tab with this file
    return redirect(url_for('dashboard', analysis_id=None) + f"?uploaded={filename}")


@app.route('/admin')
@admin_required
def admin_panel():
    # gather all analyses across users
    analyses = []
    try:
        users = db.collection('users').stream()
        for u in users:
            user_email = u.id
            analyses_ref = db.collection('users').document(user_email).collection('analyses')
            docs = analyses_ref.order_by('uploaded_at', direction=firestore.Query.DESCENDING).stream()
            for d in docs:
                data = d.to_dict()
                analyses.append({
                    'id': d.id,
                    'filename': data.get('filename'),
                    'uploaded_at': data.get('uploaded_at'),
                    'user_email': user_email,
                    'simplification_level': data.get('simplification_level')
                })
    except Exception as e:
        app.logger.warning(f"Could not load all analyses for admin panel: {e}")

    return render_template('admin.html', analyses=analyses)


@app.route('/admin/download_analysis/<user_email>/<analysis_id>')
@admin_required
def admin_download_analysis(user_email, analysis_id):
    try:
        doc_ref = db.collection('users').document(user_email).collection('analyses').document(analysis_id)
        doc = doc_ref.get()
        if not doc.exists:
            flash('Analysis not found.', 'error')
            return redirect(url_for('admin_panel'))
        data = doc.to_dict()
        rp = data.get('results_path')
        if not rp or not os.path.exists(rp):
            flash('Results file missing on server.', 'error')
            return redirect(url_for('admin_panel'))

        # load results and generate PDF similar to download_analysis
        with open(rp, 'r', encoding='utf-8') as jf:
            loaded = json.load(jf)
        if isinstance(loaded, dict) and 'results' in loaded:
            results = loaded.get('results')
            file_simp_level = loaded.get('simplification_level')
        else:
            results = loaded
            file_simp_level = None
        results = strip_clause_confidence(results)

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except Exception:
            flash('PDF generation requires the reportlab package. Install with: pip install reportlab')
            return redirect(url_for('admin_panel'))

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        margin = 72
        y = height - margin
        c.setFont('Helvetica-Bold', 16)
        header_name = data.get('filename') or 'Analysis'
        header_level = file_simp_level or data.get('simplification_level') or 'Intermediate'
        c.drawString(margin, y, f'Analysis Results — {header_name} (Simplification: {header_level})')
        y -= 24
        c.setFont('Helvetica', 10)

        def write_wrapped(text, indent=0, max_width=80):
            nonlocal y
            import textwrap
            lines = textwrap.wrap(text, max_width)
            for ln in lines:
                if y < margin + 40:
                    c.showPage()
                    y = height - margin
                    c.setFont('Helvetica', 10)
                c.drawString(margin + indent, y, ln)
                y -= 12

        for r in results:
            if y < margin + 80:
                c.showPage()
                y = height - margin
                c.setFont('Helvetica', 10)

            c.setFont('Helvetica-Bold', 12)
            c.drawString(margin, y, f"Clause {r.get('index')} — {r.get('type')}")
            y -= 14
            c.setFont('Helvetica', 10)
            write_wrapped(r.get('cleaned', ''), indent=8, max_width=90)
            y -= 6
            terms = r.get('terms') or {}
            if terms:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(margin + 8, y, 'Terms:')
                y -= 12
                c.setFont('Helvetica', 9)
                for t, info in terms.items():
                    definition = info.get('definition') if isinstance(info, dict) else info
                    method = info.get('method') if isinstance(info, dict) else ''
                    write_wrapped(f"- {t}: {definition} ({method})", indent=16, max_width=86)
                    y -= 4
            else:
                c.drawString(margin + 8, y, 'Terms: None')
                y -= 14

            c.setFont('Helvetica-Bold', 10)
            c.drawString(margin + 8, y, 'Simplified:')
            y -= 12
            c.setFont('Helvetica', 10)
            write_wrapped(r.get('simple', ''), indent=16, max_width=86)
            y -= 12

        c.save()
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"{data.get('filename')}_analysis.pdf", mimetype='application/pdf')

    except Exception as e:
        app.logger.warning(f"Admin error generating analysis PDF: {e}")
        flash('Could not generate PDF for analysis.', 'error')
        return redirect(url_for('admin_panel'))


@app.route('/admin/download_analysis_json/<user_email>/<analysis_id>')
@admin_required
def admin_download_analysis_json(user_email, analysis_id):
    try:
        doc_ref = db.collection('users').document(user_email).collection('analyses').document(analysis_id)
        doc = doc_ref.get()
        if not doc.exists:
            flash('Analysis not found.', 'error')
            return redirect(url_for('admin_panel'))
        data = doc.to_dict()
        rp = data.get('results_path')
        if not rp or not os.path.exists(rp):
            flash('Results file missing on server.', 'error')
            return redirect(url_for('admin_panel'))
        return send_file(rp, as_attachment=True, download_name=os.path.basename(rp), mimetype='application/json')
    except Exception as e:
        app.logger.warning(f"Admin error downloading analysis JSON: {e}")
        flash('Could not download analysis JSON.', 'error')
        return redirect(url_for('admin_panel'))


@app.route('/download_analysis_json/<analysis_id>')
@login_required
def download_analysis_json(analysis_id):
    try:
        doc_ref = db.collection('users').document(session.get('user_email')).collection('analyses').document(analysis_id)
        doc = doc_ref.get()
        if not doc.exists:
            flash('Analysis not found.', 'error')
            return redirect(url_for('dashboard'))
        data = doc.to_dict()
        rp = data.get('results_path')
        if not rp or not os.path.exists(rp):
            flash('Results file missing on server.', 'error')
            return redirect(url_for('dashboard'))
        return send_file(rp, as_attachment=True, download_name=os.path.basename(rp), mimetype='application/json')
    except Exception as e:
        app.logger.warning(f"Error downloading analysis JSON: {e}")
        flash('Could not download analysis.', 'error')
        return redirect(url_for('dashboard'))


@app.route('/download_analysis/<analysis_id>')
@login_required
def download_analysis(analysis_id):
    try:
        doc_ref = db.collection('users').document(session.get('user_email')).collection('analyses').document(analysis_id)
        doc = doc_ref.get()
        if not doc.exists:
            flash('Analysis not found.', 'error')
            return redirect(url_for('dashboard'))
        data = doc.to_dict()
        rp = data.get('results_path')
        if not rp or not os.path.exists(rp):
            flash('Results file missing on server.', 'error')
            return redirect(url_for('dashboard'))

        # load results file (support both legacy list and new dict wrapper)
        with open(rp, 'r', encoding='utf-8') as jf:
            loaded = json.load(jf)
        if isinstance(loaded, dict) and 'results' in loaded:
            results = loaded.get('results')
            file_simp_level = loaded.get('simplification_level') or data.get('simplification_level')
        else:
            results = loaded
            file_simp_level = data.get('simplification_level')
        results = strip_clause_confidence(results)

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except Exception:
            flash('PDF generation requires the reportlab package. Install with: pip install reportlab')
            return redirect(url_for('dashboard'))

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        margin = 72
        y = height - margin
        c.setFont('Helvetica-Bold', 16)
        header_name = data.get('filename') or 'Analysis'
        header_level = file_simp_level or data.get('simplification_level') or 'Intermediate'
        c.drawString(margin, y, f'Analysis Results — {header_name} (Simplification: {header_level})')
        y -= 24
        c.setFont('Helvetica', 10)

        def write_wrapped(text, indent=0, max_width=80):
            nonlocal y
            import textwrap
            lines = textwrap.wrap(text, max_width)
            for ln in lines:
                if y < margin + 40:
                    c.showPage()
                    y = height - margin
                    c.setFont('Helvetica', 10)
                c.drawString(margin + indent, y, ln)
                y -= 12

        for r in results:
            if y < margin + 80:
                c.showPage()
                y = height - margin
                c.setFont('Helvetica', 10)

            c.setFont('Helvetica-Bold', 12)
            c.drawString(margin, y, f"Clause {r.get('index')} — {r.get('type')}")
            y -= 14
            c.setFont('Helvetica', 10)
            write_wrapped(r.get('cleaned', ''), indent=8, max_width=90)
            y -= 6
            terms = r.get('terms') or {}
            if terms:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(margin + 8, y, 'Terms:')
                y -= 12
                c.setFont('Helvetica', 9)
                for t, info in terms.items():
                    definition = info.get('definition') if isinstance(info, dict) else info
                    method = info.get('method') if isinstance(info, dict) else ''
                    write_wrapped(f"- {t}: {definition} ({method})", indent=16, max_width=86)
                    y -= 4
            else:
                c.drawString(margin + 8, y, 'Terms: None')
                y -= 14

            c.setFont('Helvetica-Bold', 10)
            c.drawString(margin + 8, y, 'Simplified:')
            y -= 12
            c.setFont('Helvetica', 10)
            write_wrapped(r.get('simple', ''), indent=16, max_width=86)
            y -= 12

        c.save()
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"{data.get('filename')}_analysis.pdf", mimetype='application/pdf')
    except Exception as e:
        app.logger.warning(f"Error generating analysis PDF: {e}")
        flash('Could not generate PDF for analysis.', 'error')
        return redirect(url_for('dashboard'))





##===============================
#  ROUTES SECTION1: File Handling
##===============================

@app.route('/download_results')
def download_results():
    filename = request.args.get('filename')
    if not filename:
        flash('No filename provided for download')
        return redirect(url_for('dashboard'))

    # expect a results json saved as <uploaded_filename>.results.json
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + '.results.json')
    if not os.path.exists(json_path):
        flash('No results available to download for this file.')
        return redirect(url_for('dashboard'))

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        flash('PDF generation requires the reportlab package. Install with: pip install reportlab')
        return redirect(url_for('dashboard'))

    # load results (support legacy list format or new dict wrapper)
    with open(json_path, 'r', encoding='utf-8') as jf:
        loaded = json.load(jf)
    if isinstance(loaded, dict) and 'results' in loaded:
        results = loaded.get('results')
        file_simp_level = loaded.get('simplification_level')
    else:
        results = loaded
        file_simp_level = None
    results = strip_clause_confidence(results)

    # generate PDF in-memory
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 72
    y = height - margin

    # header
    c.setFont('Helvetica-Bold', 16)
    header_level = file_simp_level or ''
    if header_level:
        c.drawString(margin, y, f'Analysis Results — {filename} (Simplification: {header_level})')
    else:
        c.drawString(margin, y, f'Analysis Results — {filename}')
    y -= 24
    c.setFont('Helvetica', 10)

    def write_wrapped(text, indent=0, max_width=80):
        nonlocal y
        import textwrap
        lines = textwrap.wrap(text, max_width)
        for ln in lines:
            if y < margin + 40:
                c.showPage()
                y = height - margin
                c.setFont('Helvetica', 10)
            c.drawString(margin + indent, y, ln)
            y -= 12

    for r in results:
        if y < margin + 80:
            c.showPage()
            y = height - margin
            c.setFont('Helvetica', 10)

        c.setFont('Helvetica-Bold', 12)
        c.drawString(margin, y, f"Clause {r.get('index')} — {r.get('type')}")
        y -= 14
        c.setFont('Helvetica', 10)
        write_wrapped(r.get('cleaned', ''), indent=8, max_width=90)
        y -= 6
        # terms
        terms = r.get('terms') or {}
        if terms:
            c.setFont('Helvetica-Bold', 10)
            c.drawString(margin + 8, y, 'Terms:')
            y -= 12
            c.setFont('Helvetica', 9)
            for t, info in terms.items():
                definition = info.get('definition') if isinstance(info, dict) else info
                method = info.get('method') if isinstance(info, dict) else ''
                write_wrapped(f"- {t}: {definition} ({method})", indent=16, max_width=86)
                y -= 4
        else:
            c.drawString(margin + 8, y, 'Terms: None')
            y -= 14

        # simplified
        c.setFont('Helvetica-Bold', 10)
        c.drawString(margin + 8, y, 'Simplified:')
        y -= 12
        c.setFont('Helvetica', 10)
        write_wrapped(r.get('simple', ''), indent=16, max_width=86)
        y -= 12

    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"{filename}_analysis.pdf", mimetype='application/pdf')
#-----------------------

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)






######==============================
### TESTING 


from flask import jsonify

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'document' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['document']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Run the clause analysis pipeline
    contract_text = extract_text(save_path)
    if contract_text.startswith('[ERROR]'):
        return jsonify({"error": contract_text}), 500

    processed_clauses = preprocess_contract_text(contract_text)
    clause_types = [detect_clause_type(c['cleaned_text']) for c in processed_clauses]
    clause_terms = [recognize_legal_terms(c['cleaned_text'], legal_terms) for c in processed_clauses]
    simplified = [simplify_text(c['cleaned_text']) for c in processed_clauses]

    results = []
    for i, c in enumerate(processed_clauses):
        ct = clause_types[i]
        if isinstance(ct, (list, tuple)):
            label = ct[0]
            confidence = float(ct[1]) if len(ct) > 1 else 0.0
        else:
            label = ct
            confidence = 0.0
        results.append({
            "index": i + 1,
            "type": label or "Unknown",
            "confidence": confidence,
            "terms": clause_terms[i],
            "simplified": simplified[i],
            "text": c["cleaned_text"]
        })
    results = strip_clause_confidence(results)

    # (Optional) save in Firestore for history
    import datetime
    user_email = session.get('user_email')
    analysis_data = {
        "filename": filename,
        "uploaded_at": datetime.datetime.utcnow(),
        "results": results,
        "user_email": user_email
    }
    db.collection('users').document(user_email).collection('analyses').add(analysis_data)

    return jsonify({
        "message": "Analysis completed successfully",
        "filename": filename,
        "results_count": len(results),
        "results": results
    }), 200



##==================================
#   Flask app running at 3000
##==================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)