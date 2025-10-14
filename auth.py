# auth.py
import os
import requests
import streamlit as st

API_KEY = os.getenv("FIREBASE_API_KEY")
_SIGNIN = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
_SIGNUP = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"
_PWRESET = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={API_KEY}"

def _call(url: str, payload: dict):
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def signup(email: str, password: str):
    return _call(_SIGNUP, {"email": email, "password": password, "returnSecureToken": True})

def signin(email: str, password: str):
    return _call(_SIGNIN, {"email": email, "password": password, "returnSecureToken": True})

def send_password_reset(email: str):
    return _call(_PWRESET, {"requestType": "PASSWORD_RESET", "email": email})

def logout():
    for k in ("id_token", "email", "uid"):
        st.session_state.pop(k, None)
    st.rerun()

def require_login(title: str = "Kickflip"):
    """Renderiza login/registro si no hay sesión. Devuelve (uid, id_token, email) cuando está autenticado."""
    st.set_page_config(page_title=title)
    st.title(title)

    # Si ya hay sesión, devolver datos
    if "id_token" in st.session_state and "uid" in st.session_state:
        with st.sidebar:
            st.caption(f"Usuario: {st.session_state.get('email','')}")
            if st.button("Cerrar sesión"):
                logout()
        return st.session_state["uid"], st.session_state["id_token"], st.session_state.get("email")

    # UI de auth
    tabs = st.tabs(["Ingresar", "Registrarse", "Olvidé mi clave"])
    with tabs[0]:
        em = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        if st.button("Ingresar"):
            try:
                data = signin(em, pw)
                st.session_state["id_token"] = data["idToken"]
                st.session_state["uid"] = data["localId"]
                st.session_state["email"] = em
                st.rerun()
            except Exception:
                st.error("Credenciales inválidas")

    with tabs[1]:
        em2 = st.text_input("Email nuevo")
        pw2 = st.text_input("Password nuevo", type="password")
        if st.button("Crear cuenta"):
            try:
                signup(em2, pw2)
                st.success("Cuenta creada. Ingresá desde la pestaña 'Ingresar'.")
            except Exception:
                st.error("No se pudo crear la cuenta")

    with tabs[2]:
        em3 = st.text_input("Email para recuperar clave")
        if st.button("Enviar email de recuperación"):
            try:
                send_password_reset(em3)
                st.success("Te enviamos un email para resetear la contraseña (si el correo existe).")
            except Exception:
                st.error("No se pudo enviar el email de recuperación")

    st.stop()

def authorized_headers():
    """Devuelve headers con Bearer Token para llamar APIs protegidas."""
    tok = st.session_state.get("id_token")
    return {"Authorization": f"Bearer {tok}"} if tok else {}
