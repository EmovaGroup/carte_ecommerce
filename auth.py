import streamlit as st
from supabase_client import get_supabase

def is_logged_in() -> bool:
    return bool(st.session_state.get("sb_session"))

def logout():
    try:
        get_supabase().auth.sign_out()
    except Exception:
        pass

    st.session_state.pop("sb_session", None)
    st.session_state.pop("sb_user", None)
    st.rerun()

def login_form():
    sb = get_supabase()

    st.title("Connexion")
    with st.form("login", clear_on_submit=False):
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type="password")
        ok = st.form_submit_button("Se connecter")

    if ok:
        try:
            res = sb.auth.sign_in_with_password({"email": email, "password": password})
            session = getattr(res, "session", None) or (res.get("session") if isinstance(res, dict) else None)
            user = getattr(res, "user", None) or (res.get("user") if isinstance(res, dict) else None)

            if not session or not user:
                st.error("Identifiants invalides.")
                return

            st.session_state["sb_session"] = session
            st.session_state["sb_user"] = {
                "email": (user.email if hasattr(user, "email") else user.get("email"))
            }
            st.rerun()

        except Exception as e:
            st.error(f"Erreur : {e}")

def require_auth():
    if not is_logged_in():
        st.warning("Connecte-toi pour accéder à l'application.")
        login_form()
        st.stop()
