# Entry point – wires everything together exactly like your monolith did.

import streamlit as st
from utils import logo_data_uri
from styling import setup_page_and_css
from ui import render_app

def main():
    # Prepare logo (same behavior and warnings as before)
    LOGO_URI = logo_data_uri("assets/logo.png")

    # Page config + full CSS injection (unchanged visuals)
    setup_page_and_css(LOGO_URI)

    # Render the entire app UI
    render_app()

if __name__ == "__main__":
    main()
