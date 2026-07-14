"""
Robust Selenium login + interaction automation — a reference implementation.

Demonstrates the automation techniques a bot's "session" layer needs, done the
maintainable way:

  1. Anchor selectors on meaning (id / name / ARIA / text), not DOM position.
  2. One centralized explicit-wait model — never bare time.sleep() as a wait.
  3. Fail loud: distinguish "legitimately absent" from "page changed / broken".
  4. Page Object Model: all selectors for a screen live in one place.
  5. Verify the postcondition of every action (did the login actually succeed?)
     instead of sleeping and hoping.

Target: https://the-internet.herokuapp.com/login — a sandbox published expressly
for practicing Selenium against login forms, dynamic content, and flaky UI. It
hands out its own test credentials (tomsmith / SuperSecretPassword!), so this
file is a clean, runnable teaching artifact with no third party involved.
"""

import logging
import os
import sys

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("automation")


class LoginError(Exception):
    """Raised when a login attempt is rejected by the site (bad credentials)."""


# --- PRINCIPLE 2 + 3: one wait model, and the loud/soft distinction, in ONE place.
class Page:
    def __init__(self, driver, timeout=10):
        self.driver = driver
        self.wait = WebDriverWait(driver, timeout)

    def type_into(self, locator, text):
        el = self.wait.until(EC.visibility_of_element_located(locator))
        el.clear()
        el.send_keys(text)

    def click(self, locator):
        # wait for CLICKABLE, not merely present — an element can exist in the
        # DOM a beat before it is actually interactable.
        self.wait.until(EC.element_to_be_clickable(locator)).click()

    def text_of_required(self, locator):
        """MUST exist. If it doesn't, the page changed (or the selector rotted)
        — a bug, so fail LOUD: capture evidence, raise. Never swallow it."""
        try:
            return self.wait.until(
                EC.visibility_of_element_located(locator)).text
        except TimeoutException:
            path = "failure.png"
            self.driver.save_screenshot(path)
            log.error(
                "REQUIRED element missing — structure may have changed.\n"
                "  locator: %s\n  url: %s\n  screenshot: %s",
                locator, self.driver.current_url, path,
            )
            raise

    def find_optional(self, locator, timeout=3):
        """MAY legitimately be absent (e.g. an error banner that only appears on
        failure). Absence here is DATA, not a bug — return None quietly, but log
        it so 'not present' is always visible in the trace."""
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.visibility_of_element_located(locator))
        except TimeoutException:
            log.info("optional element absent (expected-possible): %s", locator)
            return None


# --- PRINCIPLE 1 + 4: Page Object. Every selector is a named constant up top.
class LoginPage(Page):
    URL = "https://the-internet.herokuapp.com/login"

    # Anchored on stable form semantics (id / name), not positional XPath.
    USERNAME   = (By.ID, "username")
    PASSWORD   = (By.ID, "password")
    SUBMIT     = (By.CSS_SELECTOR, 'button[type="submit"]')
    FLASH      = (By.ID, "flash")               # success/error banner
    LOGOUT_BTN = (By.CSS_SELECTOR, 'a[href="/logout"]')

    def open(self):
        self.driver.get(self.URL)
        # PRINCIPLE 5: verify the form actually rendered before touching it.
        self.wait.until(EC.visibility_of_element_located(self.USERNAME))
        log.info("login page loaded: %s", self.driver.current_url)

    def login(self, username, password):
        """Drive the auth flow, then VERIFY the postcondition. Returns the
        secure-area page object on success; raises LoginError on rejection."""
        self.type_into(self.USERNAME, username)
        self.type_into(self.PASSWORD, password)
        self.click(self.SUBMIT)

        # PRINCIPLE 5 again: don't assume the click worked — confirm we landed
        # in the secure area. The logout button is the unambiguous success mark.
        success = self.find_optional(self.LOGOUT_BTN, timeout=5)
        if success:
            log.info("login succeeded")
            return SecureAreaPage(self.driver)

        # No logout button => login did NOT succeed. Surface WHY, loudly.
        banner = self.text_of_required(self.FLASH).replace("×", "").strip()
        raise LoginError(f"login rejected: {banner!r}")


class SecureAreaPage(Page):
    """The page you reach only when authenticated — the automation's real work
    would happen here. Kept minimal; it just proves we got in and can act."""

    HEADING    = (By.CSS_SELECTOR, "div.example h2")
    LOGOUT_BTN = (By.CSS_SELECTOR, 'a[href="/logout"]')

    def heading(self):
        return self.text_of_required(self.HEADING)

    def logout(self):
        self.click(self.LOGOUT_BTN)
        # verify we actually returned to the login screen
        self.wait.until(EC.visibility_of_element_located((By.ID, "username")))
        log.info("logged out cleanly")


def build_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    # Sandboxed CI images often ship their own browser/driver pair and block
    # downloads; honor overrides there. Otherwise Selenium Manager (built in
    # since 4.6) fetches the matching driver itself.
    if os.environ.get("CHROME_BIN"):
        opts.binary_location = os.environ["CHROME_BIN"]
    service = None
    if os.environ.get("CHROMEDRIVER"):
        from selenium.webdriver.chrome.service import Service
        service = Service(executable_path=os.environ["CHROMEDRIVER"])
    return webdriver.Chrome(options=opts, service=service)


def main():
    # The sandbox publishes these credentials on the page itself.
    USER, PASSWORD = "tomsmith", "SuperSecretPassword!"

    driver = build_driver(headless=True)
    try:
        login = LoginPage(driver)
        login.open()

        # --- happy path: correct credentials ---
        secure = login.login(USER, PASSWORD)
        log.info("secure area reached: %r", secure.heading())
        secure.logout()

        # --- failure path: show that rejection fails LOUD, not silently ---
        login.open()
        try:
            login.login(USER, "wrong-password")
        except LoginError as e:
            log.info("bad-credentials path handled correctly: %s", e)

    finally:
        driver.quit()   # PRINCIPLE 3: cleanup guaranteed even on exception


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("automation run failed")   # loud, with full traceback
        sys.exit(1)
