import pymysql
pymysql.install_as_MySQLdb()
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import date, datetime, timedelta
import pickle
import numpy as np
import os
import random
import re

app = Flask(__name__)
app.secret_key = 'hoams_final_unbreakable_2026'

# DB Config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = ''
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

# Average consultation duration used for check-in time calculation
AVG_CONSULT_MIN = 15

# ═══════════════════════════════════════════════
# LOAD ML MODELS
# ═══════════════════════════════════════════════
SLOT_MODEL = None
NOSHOW_MODEL = None
FEATURE_NAMES = None

try:
    with open('models/slot_model.pkl', 'rb') as f:
        SLOT_MODEL = pickle.load(f)
    with open('models/noshow_model.pkl', 'rb') as f:
        NOSHOW_MODEL = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        FEATURE_NAMES = pickle.load(f)
    print("ML Models loaded successfully!")
except Exception as e:
    print(f"ML Models not loaded: {e}")

# ═══════════════════════════════════════════════
# EMAIL VALIDATION (.com only)
# ═══════════════════════════════════════════════
def is_valid_com_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.com$'
    return re.match(pattern, email.strip().lower()) is not None


# ═══════════════════════════════════════════════
# OTP HELPERS
# ═══════════════════════════════════════════════
def generate_otp():
    return str(random.randint(100000, 999999))


def send_otp_sms(phone, otp, purpose="verification"):
    print("\n" + "="*55)
    print(f"OTP SMS SENT TO: {phone}")
    print("-"*55)
    print(f"Your City Hospital OTP for {purpose}: {otp}")
    print(f"Valid for 10 minutes. Do not share with anyone.")
    print("="*55 + "\n")
    return True


# ═══════════════════════════════════════════════
# ML PREDICTION HELPERS
# ═══════════════════════════════════════════════
DEPT_MAPPING = {
    'General Medicine': 0, 'Orthopedics': 1, 'Pediatrics': 2,
    'Gynecology': 3, 'ENT': 4, 'Dermatology': 5,
    'Cardiology': 0, 'Neurology': 0
}

def predict_noshow_risk(patient_age, patient_gender, distance_km, is_new_patient,
                        department_name, appointment_hour, appt_date,
                        prior_noshow=0, avg_duration=10.0, appts_90d=1):
    rule_prob = 0.12

    if appointment_hour >= 13:
        rule_prob += 0.18
    elif appointment_hour >= 12:
        rule_prob += 0.10
    elif appointment_hour <= 9:
        rule_prob -= 0.03

    if is_new_patient:
        rule_prob += 0.12

    if distance_km > 25:
        rule_prob += 0.18
    elif distance_km > 15:
        rule_prob += 0.10
    elif distance_km > 8:
        rule_prob += 0.04

    rule_prob += min(0.35, prior_noshow * 0.12)

    if patient_age < 18:
        rule_prob += 0.08
    elif patient_age > 65:
        rule_prob -= 0.05

    try:
        d = datetime.strptime(str(appt_date), '%Y-%m-%d')
        dow = d.weekday()
        lead_time = (d.date() - date.today()).days

        if dow >= 5:
            rule_prob += 0.10
        if dow == 0:
            rule_prob -= 0.04

        if lead_time > 14:
            rule_prob += 0.10
        elif lead_time > 7:
            rule_prob += 0.05
        elif lead_time == 0:
            rule_prob -= 0.05
    except:
        dow, lead_time = 1, 3

    if appts_90d > 4:
        rule_prob -= 0.08

    rule_prob = max(0.05, min(0.92, rule_prob))

    if NOSHOW_MODEL is None:
        return rule_prob

    try:
        d = datetime.strptime(str(appt_date), '%Y-%m-%d')
        day_of_week = d.weekday()
        month = d.month
        lead_time = max(0, min(30, (d.date() - date.today()).days))
        is_holiday = 1 if day_of_week >= 5 else 0
        dept_id = DEPT_MAPPING.get(department_name, 0)
        visit_type = 0 if is_new_patient else 1

        features = np.array([[
            patient_age, patient_gender, distance_km, is_new_patient,
            dept_id, visit_type, appointment_hour, day_of_week,
            month, lead_time, is_holiday,
            prior_noshow, avg_duration, appts_90d
        ]])

        ml_prob = float(NOSHOW_MODEL.predict_proba(features)[0][1])
        final_prob = 0.6 * ml_prob + 0.4 * rule_prob
        return max(0.03, min(0.95, final_prob))
    except Exception as e:
        print(f"Prediction error: {e}")
        return rule_prob


def get_risk_label(prob):
    if prob >= 0.50:
        return 'High', '#dc2626'
    elif prob >= 0.28:
        return 'Medium', '#d97706'
    else:
        return 'Low', '#059669'


def get_patient_history_stats(user_id):
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total_appts,
            SUM(CASE WHEN status='No-Show' THEN 1 ELSE 0 END) as noshow_count,
            SUM(CASE WHEN appointment_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) THEN 1 ELSE 0 END) as appts_90d
        FROM appointments WHERE user_id = %s
    """, [user_id])
    res = cur.fetchone()
    cur.close()

    if not res or res['total_appts'] == 0:
        return {'is_new': 1, 'prior_noshow': 0, 'appts_90d': 0}

    return {
        'is_new': 0,
        'prior_noshow': int(res['noshow_count'] or 0),
        'appts_90d': int(res['appts_90d'] or 0)
    }


def generate_token(doctor_id, appt_date, hour):
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT COUNT(*) as cnt FROM appointments
        WHERE doctor_id = %s AND appointment_date = %s
        AND appointment_hour = %s
    """, (doctor_id, appt_date, hour))
    result = cur.fetchone()
    cur.close()
    count = result['cnt'] if result else 0
    return f"T{hour:02d}{count + 1:03d}"


# ═══════════════════════════════════════════════
# DOCTOR SLOT CONFIGURATION
# ═══════════════════════════════════════════════
def get_doctor_slots(doctor_id):
    shift_patterns = {
        0: [(9, 12), (14, 17)],
        1: [(9, 13)],
        2: [(14, 18)],
        3: [(10, 13), (15, 18)],
        4: [(8, 12), (13, 16)],
    }
    return shift_patterns.get(doctor_id % 5, [(9, 12), (14, 17)])


def get_slot_crowd(doctor_id, appt_date, start_hour, end_hour):
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT COUNT(*) as cnt FROM appointments
        WHERE doctor_id = %s AND appointment_date = %s
        AND appointment_hour >= %s AND appointment_hour < %s
        AND status IN ('Scheduled','Checked In')
    """, (doctor_id, appt_date, start_hour, end_hour))
    result = cur.fetchone()
    cur.close()
    count = result['cnt'] if result else 0

    capacity = (end_hour - start_hour) * 4

    seed = hash((doctor_id, str(appt_date), start_hour)) % 10000
    rng = random.Random(seed)

    try:
        d = datetime.strptime(str(appt_date), '%Y-%m-%d').date()
        days_ahead = (d - date.today()).days
    except:
        days_ahead = 0

    if days_ahead <= 0:
        baseline = rng.randint(int(capacity * 0.5), int(capacity * 0.9))
    elif days_ahead <= 2:
        baseline = rng.randint(int(capacity * 0.3), int(capacity * 0.7))
    elif days_ahead <= 7:
        baseline = rng.randint(int(capacity * 0.15), int(capacity * 0.5))
    else:
        baseline = rng.randint(0, int(capacity * 0.3))

    if 9 <= start_hour <= 11:
        baseline = min(capacity, baseline + 2)

    effective_count = min(capacity, count + baseline)
    fill_ratio = effective_count / capacity if capacity > 0 else 0

    if fill_ratio < 0.20:
        return ('available', effective_count, 'Slot is open', capacity)
    elif fill_ratio < 0.45:
        return ('low', effective_count, 'Light schedule expected', capacity)
    elif fill_ratio < 0.70:
        return ('medium', effective_count, 'Moderate booking', capacity)
    elif fill_ratio < 0.90:
        return ('high', effective_count, 'Slot is filling up', capacity)
    else:
        return ('full', effective_count, 'Slot nearly full', capacity)


def format_time_range(start_h, end_h):
    def fmt(h):
        if h == 0: return "12:00 AM"
        if h < 12: return f"{h}:00 AM"
        if h == 12: return "12:00 PM"
        return f"{h - 12}:00 PM"
    return f"{fmt(start_h)} - {fmt(end_h)}"


def format_time_12h(hour, minute=0):
    """Convert 24h hour:minute to '9:30 AM' style string."""
    if hour == 0:
        h = 12
        suffix = "AM"
    elif hour < 12:
        h = hour
        suffix = "AM"
    elif hour == 12:
        h = 12
        suffix = "PM"
    else:
        h = hour - 12
        suffix = "PM"
    return f"{h}:{minute:02d} {suffix}"


def get_block_for_hour(doctor_id, hour):
    """Find which slot block (start, end) an appointment hour belongs to."""
    slot_ranges = get_doctor_slots(doctor_id)
    for s, e in slot_ranges:
        if s <= hour < e:
            return s, e
    return hour, hour + 1


def compute_checkin_and_delay(doctor_id, appt_date, appt_hour, token_number, status):
    """
    Compute personalized check-in time and delay message based on
    how many patients with smaller token numbers are scheduled in
    the SAME slot block.

    Returns dict with: people_ahead, check_in_time, slot_label, delay_message
    """
    block_start, block_end = get_block_for_hour(doctor_id, appt_hour)

    cur = mysql.connection.cursor()
    if status in ('Scheduled', 'Checked In'):
        cur.execute("""
            SELECT COUNT(*) as ahead FROM appointments
            WHERE doctor_id = %s AND appointment_date = %s
              AND appointment_hour >= %s AND appointment_hour < %s
              AND status IN ('Scheduled','Checked In')
              AND token_number < %s
        """, (doctor_id, appt_date, block_start, block_end, token_number))
    else:
        cur.execute("""
            SELECT COUNT(*) as ahead FROM appointments
            WHERE doctor_id = %s AND appointment_date = %s
              AND appointment_hour >= %s AND appointment_hour < %s
              AND token_number < %s
        """, (doctor_id, appt_date, block_start, block_end, token_number))
    res = cur.fetchone()
    cur.close()
    people_ahead = res['ahead'] if res else 0

    # Personalized check-in time = block_start + (people_ahead * AVG_CONSULT_MIN)
    offset_min = people_ahead * AVG_CONSULT_MIN
    block_start_min = block_start * 60
    block_end_min = block_end * 60
    check_in_total = block_start_min + offset_min

    # Don't push beyond block end
    if check_in_total >= block_end_min:
        check_in_total = max(block_start_min, block_end_min - AVG_CONSULT_MIN)

    ci_h = check_in_total // 60
    ci_m = check_in_total % 60

    check_in_time = format_time_12h(ci_h, ci_m)
    slot_label = format_time_range(block_start, block_end)

    if people_ahead == 0:
        delay_message = "You're first in queue — please arrive on time."
        delay_short = "On time"
    else:
        low = max(0, offset_min - 5)
        high = offset_min + 10
        delay_message = f"You may face ~{low}-{high} min delay ({people_ahead} patient(s) ahead of you)."
        delay_short = f"~{offset_min} min delay"

    return {
        'people_ahead': people_ahead,
        'check_in_time': check_in_time,
        'slot_label': slot_label,
        'delay_message': delay_message,
        'delay_short': delay_short,
        'block_start': block_start,
        'block_end': block_end,
    }


def send_sms(phone_number, message):
    print("\n" + "="*55)
    print(f"SMS SENT TO: {phone_number}")
    print("-"*55)
    print(message)
    print("="*55 + "\n")
    return True


# ═══════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════
@app.route('/')
def home():
    return render_template('home.html')


# ─── REGISTRATION WITH OTP ───
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        phone = request.form.get('phone', '').strip()
        age = int(request.form.get('age', 25))
        gender = int(request.form.get('gender', 0))
        distance_km = float(request.form.get('distance_km', 10.0))

        if not is_valid_com_email(email):
            flash("Only .com email addresses are allowed (e.g. user@gmail.com)", "danger")
            return render_template('register.html')

        if not phone or not phone.isdigit() or len(phone) != 10:
            flash("Please provide a valid 10-digit mobile number for OTP verification.", "danger")
            return render_template('register.html')

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM users WHERE email = %s", [email])
            if cur.fetchone():
                flash("Email already registered. Please login.", "warning")
                cur.close()
                return redirect(url_for('login'))
            cur.close()

            otp = generate_otp()
            session['pending_registration'] = {
                'name': name,
                'email': email,
                'password': generate_password_hash(password),
                'phone': phone,
                'age': age,
                'gender': gender,
                'distance_km': distance_km,
                'otp': otp,
                'otp_created': datetime.now().isoformat(),
                'attempts': 0
            }
            send_otp_sms(phone, otp, "account registration")
            flash(f"OTP sent to +91-{phone}. Please verify to complete registration.", "success")
            return redirect(url_for('verify_register_otp'))

        except Exception as e:
            print(f"Registration Error: {e}")
            flash(f"An error occurred: {e}", "danger")

    return render_template('register.html')


@app.route('/verify_register_otp', methods=['GET', 'POST'])
def verify_register_otp():
    pending = session.get('pending_registration')
    if not pending:
        flash("Session expired. Please register again.", "warning")
        return redirect(url_for('register'))

    try:
        created = datetime.fromisoformat(pending['otp_created'])
        if (datetime.now() - created).total_seconds() > 600:
            session.pop('pending_registration', None)
            flash("OTP expired. Please register again.", "danger")
            return redirect(url_for('register'))
    except:
        pass

    if request.method == 'POST':
        entered = request.form.get('otp', '').strip()
        pending['attempts'] = pending.get('attempts', 0) + 1
        session['pending_registration'] = pending

        if pending['attempts'] > 5:
            session.pop('pending_registration', None)
            flash("Too many wrong attempts. Please register again.", "danger")
            return redirect(url_for('register'))

        if entered == pending['otp']:
            try:
                cur = mysql.connection.cursor()
                cur.execute("""
                    INSERT INTO users (name, email, password, role, phone, age, gender, distance_km)
                    VALUES (%s, %s, %s, 'patient', %s, %s, %s, %s)
                """, (pending['name'], pending['email'], pending['password'],
                      pending['phone'], pending['age'], pending['gender'], pending['distance_km']))
                mysql.connection.commit()
                cur.close()
                session.pop('pending_registration', None)
                flash("Account verified and created! Please login.", "success")
                return redirect(url_for('login'))
            except Exception as e:
                flash(f"Database error: {e}", "danger")
        else:
            remaining = 5 - pending['attempts']
            flash(f"Invalid OTP. {remaining} attempts left.", "danger")

    return render_template('verify_otp.html',
                           phone=pending['phone'],
                           purpose="Account Registration",
                           resend_url=url_for('resend_register_otp'))


@app.route('/resend_register_otp')
def resend_register_otp():
    pending = session.get('pending_registration')
    if not pending:
        return redirect(url_for('register'))
    new_otp = generate_otp()
    pending['otp'] = new_otp
    pending['otp_created'] = datetime.now().isoformat()
    pending['attempts'] = 0
    session['pending_registration'] = pending
    send_otp_sms(pending['phone'], new_otp, "account registration")
    flash("New OTP sent successfully.", "success")
    return redirect(url_for('verify_register_otp'))


# ─── LOGIN ───
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email').strip().lower()
        pw = request.form.get('password').strip()

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", [email])
        user = cur.fetchone()
        cur.close()

        if user:
            db_password = user['password']
            db_role = user['role'].lower() if user['role'] else 'patient'

            is_valid = False
            if db_password == pw:
                is_valid = True
            else:
                try:
                    is_valid = check_password_hash(db_password, pw)
                except Exception:
                    is_valid = False

            if is_valid:
                session.clear()
                session.update({
                    'loggedin': True,
                    'id': user['id'],
                    'name': user['name'],
                    'role': db_role,
                    'is_doctor': (db_role == 'doctor')
                })

                if db_role == 'admin':
                    return redirect(url_for('admin_dashboard'))
                elif db_role == 'doctor':
                    return redirect(url_for('doctor_dashboard'))
                else:
                    return redirect(url_for('patient_dashboard'))

        flash("Invalid Email or Password", "danger")

    return render_template('login.html')


@app.route('/patient_dashboard')
def patient_dashboard():
    if not session.get('loggedin') or session.get('role') != 'patient':
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT a.*, d.name AS doc_name, d.specialization
        FROM appointments a
        LEFT JOIN doctors d ON a.doctor_id = d.id
        WHERE a.user_id = %s
        ORDER BY a.appointment_date DESC, a.appointment_hour DESC
    """, [session['id']])
    appts = cur.fetchall()

    cur.execute("SELECT * FROM users WHERE id = %s", [session['id']])
    user = cur.fetchone()
    cur.close()

    history = get_patient_history_stats(session['id'])

    today_str = str(date.today())
    today_appt = None
    stats = {'total': len(appts), 'upcoming': 0, 'completed': 0, 'cancelled': 0}

    for appt in appts:
        # ── Personalized check-in time and delay message
        try:
            info = compute_checkin_and_delay(
                appt['doctor_id'],
                appt['appointment_date'],
                appt['appointment_hour'],
                appt['token_number'],
                appt['status']
            )
            appt['people_ahead'] = info['people_ahead']
            appt['check_in_time'] = info['check_in_time']
            appt['slot_label'] = info['slot_label']
            appt['delay_message'] = info['delay_message']
            appt['delay_short'] = info['delay_short']
        except Exception as e:
            print(f"Check-in calc error: {e}")
            appt['people_ahead'] = 0
            appt['check_in_time'] = format_time_12h(appt['appointment_hour'], 0)
            appt['slot_label'] = format_time_range(appt['appointment_hour'], appt['appointment_hour'] + 1)
            appt['delay_message'] = "Please arrive on time."
            appt['delay_short'] = "On time"

        # ── Risk prediction
        if user:
            try:
                prob = predict_noshow_risk(
                    patient_age=user.get('age', 30) or 30,
                    patient_gender=user.get('gender', 0) or 0,
                    distance_km=user.get('distance_km', 10) or 10,
                    is_new_patient=history['is_new'],
                    department_name=appt.get('dept', 'General Medicine'),
                    appointment_hour=appt['appointment_hour'],
                    appt_date=appt['appointment_date'],
                    prior_noshow=history['prior_noshow'],
                    appts_90d=history['appts_90d']
                )
                risk_label, color = get_risk_label(prob)
                appt['noshow_risk'] = risk_label
                appt['noshow_color'] = color
                appt['noshow_prob'] = round(prob * 100, 1)
            except Exception as e:
                print(f"Risk error: {e}")
                appt['noshow_risk'] = 'Low'
                appt['noshow_color'] = '#059669'
                appt['noshow_prob'] = 15.0
        else:
            appt['noshow_risk'] = 'Low'
            appt['noshow_color'] = '#059669'
            appt['noshow_prob'] = 15.0

        if appt['status'] == 'Scheduled':
            stats['upcoming'] += 1
            if str(appt['appointment_date']) == today_str:
                today_appt = appt
        elif appt['status'] == 'Completed':
            stats['completed'] += 1
        elif appt['status'] == 'Cancelled':
            stats['cancelled'] += 1

    return render_template('patient_dashboard.html', appointments=appts, today_appt=today_appt, stats=stats)


@app.route('/appointment_history')
def appointment_history():
    if not session.get('loggedin'):
        return redirect(url_for('login'))
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT a.*, d.name AS doc_name
        FROM appointments a
        LEFT JOIN doctors d ON a.doctor_id = d.id
        WHERE a.user_id = %s
        ORDER BY a.appointment_date DESC
    """, [session['id']])
    appts = cur.fetchall()
    cur.close()

    for appt in appts:
        try:
            info = compute_checkin_and_delay(
                appt['doctor_id'],
                appt['appointment_date'],
                appt['appointment_hour'],
                appt['token_number'],
                appt['status']
            )
            appt['people_ahead'] = info['people_ahead']
            appt['check_in_time'] = info['check_in_time']
            appt['slot_label'] = info['slot_label']
            appt['delay_message'] = info['delay_message']
            appt['delay_short'] = info['delay_short']
        except:
            appt['people_ahead'] = 0
            appt['check_in_time'] = format_time_12h(appt['appointment_hour'], 0)
            appt['slot_label'] = format_time_range(appt['appointment_hour'], appt['appointment_hour'] + 1)
            appt['delay_message'] = ""
            appt['delay_short'] = ""

        appt['noshow_risk'] = 'Low'
        appt['noshow_color'] = '#059669'

    stats = {
        'total': len(appts),
        'upcoming': sum(1 for a in appts if a['status'] == 'Scheduled'),
        'completed': sum(1 for a in appts if a['status'] == 'Completed'),
        'cancelled': sum(1 for a in appts if a['status'] == 'Cancelled')
    }
    return render_template('patient_dashboard.html', appointments=appts, today_appt=None, stats=stats)


# ─── BOOKING WITH OTP ───
@app.route('/book', methods=['GET', 'POST'])
def book_appointment():
    if not session.get('loggedin'):
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        dept = request.form.get('department')
        doc_id = request.form.get('doctor_id')
        appt_date = request.form.get('date')
        hour = int(request.form.get('hour'))

        cur.execute("SELECT phone, name FROM users WHERE id = %s", [session['id']])
        user = cur.fetchone()
        cur.close()

        if not user or not user.get('phone'):
            flash("Phone number missing in your profile. Cannot send OTP.", "danger")
            return redirect(url_for('book_appointment'))

        otp = generate_otp()
        session['pending_booking'] = {
            'dept': dept,
            'doc_id': doc_id,
            'appt_date': appt_date,
            'hour': hour,
            'otp': otp,
            'otp_created': datetime.now().isoformat(),
            'attempts': 0,
            'phone': user['phone'],
            'name': user['name']
        }
        send_otp_sms(user['phone'], otp, "appointment booking")
        flash(f"OTP sent to +91-{user['phone']} to confirm your appointment.", "success")
        return redirect(url_for('verify_booking_otp'))

    cur.execute("SELECT * FROM doctors ORDER BY specialization")
    doctors = cur.fetchall()
    cur.close()
    return render_template('book.html', doctors=doctors)


@app.route('/verify_booking_otp', methods=['GET', 'POST'])
def verify_booking_otp():
    if not session.get('loggedin'):
        return redirect(url_for('login'))

    pending = session.get('pending_booking')
    if not pending:
        flash("No pending booking. Please start over.", "warning")
        return redirect(url_for('book_appointment'))

    try:
        created = datetime.fromisoformat(pending['otp_created'])
        if (datetime.now() - created).total_seconds() > 600:
            session.pop('pending_booking', None)
            flash("OTP expired. Please book again.", "danger")
            return redirect(url_for('book_appointment'))
    except:
        pass

    if request.method == 'POST':
        entered = request.form.get('otp', '').strip()
        pending['attempts'] = pending.get('attempts', 0) + 1
        session['pending_booking'] = pending

        if pending['attempts'] > 5:
            session.pop('pending_booking', None)
            flash("Too many wrong attempts. Please book again.", "danger")
            return redirect(url_for('book_appointment'))

        if entered == pending['otp']:
            cur = mysql.connection.cursor()
            cur.execute("SELECT name FROM doctors WHERE id = %s", [pending['doc_id']])
            doctor = cur.fetchone()
            doctor_name = doctor['name'] if doctor else "Staff"

            token = generate_token(pending['doc_id'], pending['appt_date'], pending['hour'])

            cur.execute("""
                INSERT INTO appointments(
                    user_id, dept, doctor_id, doctor_name,
                    appointment_date, appointment_hour, status, token_number
                )
                VALUES(%s, %s, %s, %s, %s, %s, 'Scheduled', %s)
            """, (session['id'], pending['dept'], pending['doc_id'], doctor_name,
                  pending['appt_date'], pending['hour'], token))
            mysql.connection.commit()

            cur.execute("SELECT phone, name, age, gender, distance_km FROM users WHERE id = %s", [session['id']])
            user = cur.fetchone()
            cur.close()

            history = get_patient_history_stats(session['id'])
            risk_label = "Low"
            if user:
                try:
                    risk_prob = predict_noshow_risk(
                        patient_age=user.get('age', 30) or 30,
                        patient_gender=user.get('gender', 0) or 0,
                        distance_km=user.get('distance_km', 10) or 10,
                        is_new_patient=history['is_new'],
                        department_name=pending['dept'],
                        appointment_hour=pending['hour'],
                        appt_date=pending['appt_date'],
                        prior_noshow=history['prior_noshow'],
                        appts_90d=history['appts_90d']
                    )
                    risk_label, _ = get_risk_label(risk_prob)
                except:
                    pass

            # Compute personalized check-in time for SMS
            try:
                info = compute_checkin_and_delay(
                    int(pending['doc_id']),
                    pending['appt_date'],
                    pending['hour'],
                    token,
                    'Scheduled'
                )
                check_in_time = info['check_in_time']
                slot_label = info['slot_label']
                delay_msg = info['delay_message']
            except:
                check_in_time = format_time_12h(pending['hour'], 0)
                slot_label = format_time_range(pending['hour'], pending['hour'] + 1)
                delay_msg = ""

            if user and user.get('phone'):
                msg = (
                    f"Hi {user['name']}, your appointment is CONFIRMED!\n"
                    f"Token: {token}\n"
                    f"Department: {pending['dept']}\n"
                    f"Doctor: {doctor_name}\n"
                    f"Date: {pending['appt_date']}\n"
                    f"Slot: {slot_label}\n"
                    f"Your Check-in: {check_in_time}\n"
                    f"{delay_msg}\n"
                    f"Reliability: {risk_label}\n"
                    f"- City Hospital HOAMS"
                )
                send_sms(user['phone'], msg)

            session.pop('pending_booking', None)
            flash(f"Appointment confirmed! Token: {token} | Check-in: {check_in_time}", "success")
            return redirect(url_for('patient_dashboard'))
        else:
            remaining = 5 - pending['attempts']
            flash(f"Invalid OTP. {remaining} attempts left.", "danger")

    return render_template('verify_otp.html',
                           phone=pending['phone'],
                           purpose="Appointment Booking",
                           resend_url=url_for('resend_booking_otp'))


@app.route('/resend_booking_otp')
def resend_booking_otp():
    pending = session.get('pending_booking')
    if not pending:
        return redirect(url_for('book_appointment'))
    new_otp = generate_otp()
    pending['otp'] = new_otp
    pending['otp_created'] = datetime.now().isoformat()
    pending['attempts'] = 0
    session['pending_booking'] = pending
    send_otp_sms(pending['phone'], new_otp, "appointment booking")
    flash("New OTP sent.", "success")
    return redirect(url_for('verify_booking_otp'))


# ─── DOCTOR DASHBOARD ───
@app.route('/doctor_dashboard')
def doctor_dashboard():
    if not session.get('loggedin') or session.get('role') != 'doctor':
        return redirect(url_for('login'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM doctors WHERE user_id = %s OR name = %s",
                    (session.get('id'), session.get('name')))
        doc = cur.fetchone()

        if not doc:
            cur.close()
            flash("Doctor profile not found.", "danger")
            return redirect(url_for('login'))

        today_str = str(date.today())
        cur.execute("""
            SELECT a.*, u.name AS patient_name, u.age, u.gender, u.phone
            FROM appointments a
            LEFT JOIN users u ON a.user_id = u.id
            WHERE a.doctor_id = %s
              AND DATE(a.appointment_date) = %s
            ORDER BY a.appointment_hour ASC, a.token_number ASC
        """, (doc['id'], today_str))
        queue = cur.fetchall()
        cur.close()

        today_stats = {
            'total': len(queue),
            'scheduled': sum(1 for q in queue if q['status'] == 'Scheduled'),
            'checked_in': sum(1 for q in queue if q['status'] == 'Checked In'),
            'completed': sum(1 for q in queue if q['status'] == 'Completed'),
            'no_show': sum(1 for q in queue if q['status'] == 'No-Show'),
        }

        return render_template('doctor_dashboard.html',
                               queue=queue, date=date, doctor=doc,
                               today_stats=today_stats, today_str=today_str)

    except Exception as e:
        print(f"Doctor Dashboard Error: {e}")
        return f"Dashboard Error: {e}"


@app.route('/update_status/<int:appt_id>/<string:status>')
def update_status(appt_id, status):
    if not session.get('loggedin'):
        return redirect(url_for('login'))
    cur = mysql.connection.cursor()
    cur.execute("UPDATE appointments SET status = %s WHERE appointment_id = %s", (status, appt_id))
    mysql.connection.commit()
    cur.close()
    flash(f"Patient marked as {status}", "success")
    return redirect(request.referrer or url_for('doctor_dashboard'))


# ─── ADMIN DASHBOARD ───
@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    today_str = str(date.today())

    cur.execute("SELECT COUNT(*) as c FROM appointments WHERE appointment_date = %s", [today_str])
    total_today = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM appointments WHERE appointment_date = %s AND status='Scheduled'", [today_str])
    sched = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM appointments WHERE appointment_date = %s AND status='Completed'", [today_str])
    done = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM appointments WHERE appointment_date = %s AND status='No-Show'", [today_str])
    ns = cur.fetchone()['c']

    stats = {
        'total': total_today, 'scheduled': sched, 'completed': done,
        'no_show': ns, 'avg_wait': 12,
        'risk': {'High': 0, 'Medium': 0, 'Low': 0}
    }

    cur.execute("""
        SELECT a.*, u.name AS patient_name, u.age, u.gender, u.distance_km,
               d.name AS doctor_name, d.specialization AS department_name
        FROM appointments a
        LEFT JOIN users u ON a.user_id = u.id
        LEFT JOIN doctors d ON a.doctor_id = d.id
        ORDER BY a.appointment_id DESC LIMIT 8
    """)
    recent_appts = cur.fetchall()

    for a in recent_appts:
        a['department_id'] = DEPT_MAPPING.get(a.get('dept', ''), 0)
        try:
            history = get_patient_history_stats(a['user_id']) if a.get('user_id') else {'is_new': 1, 'prior_noshow': 0, 'appts_90d': 0}
            prob = predict_noshow_risk(
                patient_age=a.get('age', 30) or 30,
                patient_gender=a.get('gender', 0) or 0,
                distance_km=a.get('distance_km', 10) or 10,
                is_new_patient=history['is_new'],
                department_name=a.get('dept', 'General Medicine'),
                appointment_hour=a['appointment_hour'],
                appt_date=a['appointment_date'],
                prior_noshow=history['prior_noshow'],
                appts_90d=history['appts_90d']
            )
            lbl, col = get_risk_label(prob)
            a['noshow_risk'] = lbl
            a['noshow_color'] = col
            a['noshow_prob'] = round(prob * 100, 1)
            stats['risk'][lbl] += 1
        except Exception as e:
            print(f"Admin risk error: {e}")
            a['noshow_risk'] = 'Low'
            a['noshow_color'] = '#059669'
            a['noshow_prob'] = 15.0
            stats['risk']['Low'] += 1

    cur.execute("""
        SELECT dept AS department_name, COUNT(*) AS cnt
        FROM appointments WHERE appointment_date = %s
        GROUP BY dept ORDER BY cnt DESC
    """, [today_str])
    dept_data = cur.fetchall()

    cur.execute("""
        SELECT d.name, d.specialization,
               (SELECT COUNT(*) FROM appointments a
                WHERE a.doctor_id = d.id
                AND a.appointment_date = %s
                AND a.status='Scheduled') AS active_count
        FROM doctors d
        ORDER BY active_count DESC LIMIT 6
    """, [today_str])
    doc_load = cur.fetchall()

    logs = []
    cur.close()

    return render_template('admin_dashboard.html', stats=stats, recent_appts=recent_appts,
                           dept_data=dept_data, doc_load=doc_load, logs=logs)


@app.route('/admin/update_status/<int:appt_id>/<string:status>', methods=['POST', 'GET'])
def admin_update_status(appt_id, status):
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    cur = mysql.connection.cursor()
    cur.execute("UPDATE appointments SET status = %s WHERE appointment_id = %s", (status, appt_id))
    mysql.connection.commit()
    cur.close()
    flash(f"Appointment marked as {status}", "success")
    return redirect(url_for('admin_dashboard'))


@app.route('/api/live_stats')
def api_live_stats():
    if session.get('role') != 'admin':
        return jsonify({'error': 'unauthorized'}), 401
    cur = mysql.connection.cursor()
    today_str = str(date.today())
    cur.execute("SELECT COUNT(*) as c FROM appointments WHERE appointment_date = %s", [today_str])
    total = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM appointments WHERE appointment_date = %s AND status='Scheduled'", [today_str])
    sched = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM appointments WHERE appointment_date = %s AND status='Completed'", [today_str])
    done = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM appointments WHERE appointment_date = %s AND status='No-Show'", [today_str])
    ns = cur.fetchone()['c']
    cur.close()
    return jsonify({
        'total': total, 'scheduled': sched, 'completed': done, 'no_show': ns,
        'avg_wait': 12, 'risk': {'High': 0, 'Medium': 0, 'Low': 0}
    })


@app.route('/api/get_slots', methods=['POST'])
def get_slots():
    if not session.get('loggedin'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    doctor_id = int(data.get('doctor_id'))
    appt_date = data.get('date')

    if not doctor_id or not appt_date:
        return jsonify({'error': 'Missing data'}), 400

    slot_ranges = get_doctor_slots(doctor_id)
    slots = []

    for start_h, end_h in slot_ranges:
        level, count, message, capacity = get_slot_crowd(doctor_id, appt_date, start_h, end_h)
        slots.append({
            'hour': start_h,
            'end_hour': end_h,
            'label': format_time_range(start_h, end_h),
            'level': level,
            'count': count,
            'capacity': capacity,
            'message': message,
            'fill_percent': round((count / capacity * 100) if capacity > 0 else 0, 0)
        })

    return jsonify({'slots': slots})


@app.route('/reports')
def reports():
    if not session.get('loggedin'):
        return redirect(url_for('login'))
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT a.*, d.name AS doc_name, d.specialization
        FROM appointments a
        LEFT JOIN doctors d ON a.doctor_id = d.id
        WHERE a.user_id = %s AND a.status = 'Completed'
        ORDER BY a.appointment_date DESC
    """, [session['id']])
    completed = cur.fetchall()
    cur.close()

    reports_list = []
    for appt in completed:
        reports_list.append({
            'id': appt['appointment_id'],
            'title': f"{appt['dept']} Consultation Report",
            'doctor': appt.get('doc_name', 'Staff'),
            'date': appt['appointment_date'],
            'type': 'Consultation Summary',
            'status': 'Available'
        })

    if len(reports_list) < 3:
        sample_reports = [
            {'id': 'BLD001', 'title': 'Complete Blood Count (CBC)', 'doctor': 'Dr. Mehta',
             'date': '2025-01-15', 'type': 'Lab Report', 'status': 'Available'},
            {'id': 'XR002', 'title': 'Chest X-Ray Report', 'doctor': 'Dr. Sharma',
             'date': '2024-12-20', 'type': 'Radiology', 'status': 'Available'},
            {'id': 'ECG003', 'title': 'ECG Analysis', 'doctor': 'Dr. Kapoor',
             'date': '2024-11-10', 'type': 'Cardiology', 'status': 'Available'},
        ]
        reports_list.extend(sample_reports)

    return render_template('reports.html', reports=reports_list)


@app.route('/departments')
def departments():
    if not session.get('loggedin'):
        return redirect(url_for('login'))
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM doctors ORDER BY specialization")
    doctors = cur.fetchall()
    cur.close()

    dept_info = {
        'Cardiology': {'icon': '❤️', 'desc': 'Heart & cardiovascular care', 'color': '#ef4444'},
        'Neurology': {'icon': '🧠', 'desc': 'Brain & nervous system', 'color': '#8b5cf6'},
        'Orthopedics': {'icon': '🦴', 'desc': 'Bones, joints & muscles', 'color': '#f59e0b'},
        'General Medicine': {'icon': '🩺', 'desc': 'Primary healthcare', 'color': '#10b981'},
        'Pediatrics': {'icon': '👶', 'desc': 'Children\'s health', 'color': '#3b82f6'},
        'Dermatology': {'icon': '🔬', 'desc': 'Skin & hair care', 'color': '#ec4899'},
        'ENT': {'icon': '👂', 'desc': 'Ear, nose & throat', 'color': '#06b6d4'},
        'Gynecology': {'icon': '🌸', 'desc': 'Women\'s health', 'color': '#f43f5e'},
    }

    departments_data = {}
    for doc in doctors:
        spec = doc['specialization']
        if spec not in departments_data:
            departments_data[spec] = {
                'info': dept_info.get(spec, {'icon': '⚕️', 'desc': 'Specialist care', 'color': '#64748b'}),
                'doctors': []
            }
        departments_data[spec]['doctors'].append(doc)

    return render_template('departments.html', departments=departments_data)


@app.route('/cancel/<int:id>', methods=['POST'])
def cancel_appointment(id):
    if not session.get('loggedin'):
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT a.*, u.phone, u.name as user_name
        FROM appointments a
        LEFT JOIN users u ON a.user_id = u.id
        WHERE a.appointment_id = %s
    """, [id])
    appt = cur.fetchone()

    cur.execute("UPDATE appointments SET status = 'Cancelled' WHERE appointment_id = %s", [id])
    mysql.connection.commit()
    cur.close()

    if appt and appt.get('phone'):
        sms_message = (
            f"Hi {appt['user_name']}, your appointment has been cancelled.\n"
            f"Department: {appt['dept']}\n"
            f"Date: {appt['appointment_date']} at {appt['appointment_hour']}:00\n"
            f"Book again anytime via HOAMS.\n"
            f"- City Hospital"
        )
        send_sms(appt['phone'], sms_message)

    flash("Appointment cancelled. SMS sent.", "success")
    return redirect(url_for('patient_dashboard'))


@app.route('/reschedule/<int:id>', methods=['GET', 'POST'])
def reschedule_appointment(id):
    if not session.get('loggedin'):
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT a.*, d.name AS doc_name, d.specialization, u.phone, u.name AS user_name
        FROM appointments a
        LEFT JOIN doctors d ON a.doctor_id = d.id
        LEFT JOIN users u ON a.user_id = u.id
        WHERE a.appointment_id = %s AND a.user_id = %s
    """, (id, session['id']))
    appt = cur.fetchone()

    if not appt:
        cur.close()
        flash("Appointment not found.", "danger")
        return redirect(url_for('patient_dashboard'))

    if appt['status'] != 'Scheduled':
        cur.close()
        flash("Only scheduled appointments can be rescheduled.", "danger")
        return redirect(url_for('patient_dashboard'))

    if request.method == 'POST':
        new_date = request.form.get('new_date')
        new_hour = request.form.get('new_hour')

        cur.execute("""
            UPDATE appointments
            SET appointment_date = %s, appointment_hour = %s
            WHERE appointment_id = %s
        """, (new_date, new_hour, id))
        mysql.connection.commit()
        cur.close()

        if appt.get('phone'):
            sms_message = (
                f"Hi {appt['user_name']}, your appointment has been rescheduled!\n"
                f"Department: {appt['dept']}\n"
                f"Doctor: {appt.get('doc_name', 'Staff')}\n"
                f"OLD: {appt['appointment_date']} at {appt['appointment_hour']}:00\n"
                f"NEW: {new_date} at {new_hour}:00\n"
                f"- City Hospital HOAMS"
            )
            send_sms(appt['phone'], sms_message)

        flash("Appointment rescheduled! SMS sent.", "success")
        return redirect(url_for('patient_dashboard'))

    cur.close()
    return render_template('reschedule_new.html', appt=appt)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/ml_demo')
def ml_demo():
    if not session.get('loggedin'):
        return redirect(url_for('login'))

    try:
        import pandas as pd
        df = pd.read_csv('appointments_dataset.csv')
        samples = df.sample(n=10, random_state=random.randint(1, 1000))

        predictions = []
        for _, row in samples.iterrows():
            prob = predict_noshow_risk(
                patient_age=int(row['patient_age']),
                patient_gender=int(row['patient_gender']),
                distance_km=float(row['distance_km']),
                is_new_patient=int(row['is_new_patient']),
                department_name=row['department_name'],
                appointment_hour=int(row['appointment_hour']),
                appt_date=date.today() + timedelta(days=int(row['lead_time_days'])),
                prior_noshow=int(row['prior_noshow_count']),
                avg_duration=float(row['avg_past_duration']),
                appts_90d=int(row['appointments_90d'])
            )
            risk_label, color = get_risk_label(prob)

            predictions.append({
                'patient_id': int(row['patient_id']),
                'age': int(row['patient_age']),
                'gender': 'Male' if row['patient_gender'] == 0 else 'Female',
                'department': row['department_name'],
                'distance': round(float(row['distance_km']), 1),
                'is_new': 'New' if row['is_new_patient'] == 1 else 'Returning',
                'hour': int(row['appointment_hour']),
                'lead_days': int(row['lead_time_days']),
                'prior_noshow': int(row['prior_noshow_count']),
                'predicted_prob': round(prob * 100, 1),
                'predicted_risk': risk_label,
                'color': color,
                'actual_outcome': 'No-Show' if row['no_show'] == 1 else 'Showed Up',
                'match': 'Correct' if (prob >= 0.5) == (row['no_show'] == 1) else 'Wrong'
            })

        correct = sum(1 for p in predictions if p['match'] == 'Correct')
        accuracy = round(correct / len(predictions) * 100, 1)

        return render_template('ml_demo.html',
                             predictions=predictions,
                             accuracy=accuracy,
                             total=len(predictions),
                             correct=correct)
    except Exception as e:
        flash(f"Error loading dataset: {e}", "danger")
        return redirect(url_for('patient_dashboard'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
