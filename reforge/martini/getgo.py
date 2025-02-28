import argparse
import os
import platform
import shutil
import subprocess as sp
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from reforge.utils import logger


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Automate downloading and extracting Go maps from PDB files."
    )
    parser.add_argument(
        "-d", "--directory", required=True, help="Directory containing PDB files."
    )
    parser.add_argument("-f", "--file", required=True, help="Name of the PDB file.")
    return parser.parse_args()


def check_browser(browser_name, command):
    """
    Check if a browser is installed by looking for its executable command.
    """
    return shutil.which(command) is not None


def check_debian_package(package_name):
    """
    Check if a Debian package is installed.
    """
    try:
        result = sp.run(["dpkg", "-s", package_name], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_webdriver(browser_name):
    """
    Install the WebDriver for the specified browser.
    """
    try:
        if browser_name == "chrome":
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service as ChromeService

            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service)
        elif browser_name == "firefox":
            from webdriver_manager.firefox import GeckoDriverManager
            from selenium.webdriver.firefox.service import Service as FirefoxService

            service = FirefoxService(GeckoDriverManager().install())
            driver = webdriver.Firefox(service=service)
        elif browser_name == "edge":
            from webdriver_manager.microsoft import EdgeChromiumDriverManager
            from selenium.webdriver.edge.service import Service as EdgeService

            service = EdgeService(EdgeChromiumDriverManager().install())
            driver = webdriver.Edge(service=service)
        elif browser_name == "safari" and platform.system() == "Darwin":
            driver = webdriver.Safari()
        else:
            logger.info(
                f"WebDriver installation for {browser_name} is not supported in this script."
            )
            return

        logger.info(f"WebDriver for {browser_name} installed successfully.")
        driver.quit()
    except Exception as e:
        logger.info(f"Error installing WebDriver for {browser_name}: {str(e)}")


def check_browsers():
    """
    Check for installed browsers and logger.info their status.
    """
    browsers = {
        "firefox": check_browser("Firefox", "firefox")
        or check_debian_package("firefox-esr"),
        "chrome": check_browser("Chrome", "google-chrome")
        or check_browser("Chromium", "chromium-browser"),
        "edge": check_browser("Edge", "microsoft-edge"),
        "safari": platform.system() == "Darwin" and check_browser("Safari", "safari"),
    }

    logger.info("Installed browsers:")
    for browser, installed in browsers.items():
        logger.info(
            f"{browser.capitalize()}: {'Installed' if installed else 'Not installed'}"
        )


def check_geckodriver_installed():
    try:
        # Try to run 'geckodriver --version' and capture output
        sp.run(["geckodriver", "--version"], capture_output=True, text=True, check=True)
        return True  # Geckodriver is installed
    except sp.CalledProcessError:
        return False  # Geckodriver command failed, but it exists
    except FileNotFoundError:
        return False  # Geckodriver is not installed or not in PATH


def init_webdriver(download_dir):
    """
    Initialize the Firefox WebDriver with specified options.
    """
    logger.info("Initializing WebDriver...")
    options = Options()
    options.add_argument(
        "-headless"
    )  # Run in headless mode, remove this if you want to see the browser

    # Set preferences for download behavior
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir", os.path.abspath(download_dir))
    options.set_preference(
        "browser.helperApps.neverAsk.saveToDisk", "application/x-gzip"
    )

    # Initialize the web driver
    if check_geckodriver_installed():
        driver = webdriver.Firefox(options=options)
    else:
        driver = webdriver.Firefox(
            service=FirefoxService(GeckoDriverManager().install()), options=options
        )
    logger.info("WebDriver initialized.")
    return driver


def get_go_maps(driver, pdb_files):
    """
    Use Selenium to automate downloading Go maps from the server.
    pdb_files - path to pdb files
    """
    logger.info("Submitting PDBs...")

    for f in pdb_files:
        logger.info(f"Processing {f}...")
        driver.get("http://info.ifpan.edu.pl/~rcsu/rcsu/index.html")

        try:
            pdb_input = driver.find_element(By.NAME, "filename")
            pdb_input.send_keys(f)

            driver.find_element(By.XPATH, "//input[@type='SUBMIT']").click()

            download_link = WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, "here"))
            )

            logger.info("Downloading the map...")
            download_link.click()

            time.sleep(10)  # Wait for the download to complete
            logger.info(f"Downloaded Go map for {f}.")

        except Exception as e:
            logger.info(f"An error occurred while processing {f}: {str(e)}")

    driver.quit()
    logger.info("Go maps download process completed.")


def extract_go_maps(wdir):
    """
    Extract downloaded Go maps from tar.gz files and organize them.
    """
    logger.info("Extracting Go maps...")
    tgz_files = [f for f in os.listdir(wdir) if f.endswith(".tgz")]
    for f in tgz_files:
        tgz_path = os.path.join(wdir, f)
        logger.info(f"Extracting {f}...")
        sp.run(["tar", "-xzf", tgz_path, "-C", wdir])
        os.remove(tgz_path)

    work2_dir = os.path.join(wdir, "work2")
    if os.path.exists(work2_dir):
        dirs = [d for d in os.listdir(work2_dir)]
        for d in dirs:
            source_dir = os.path.join(work2_dir, d)
            files = os.listdir(source_dir)
            if files:
                source_file = os.path.join(source_dir, files[0])
                shutil.move(source_file, os.path.join(wdir, files[0]))
        shutil.rmtree(work2_dir)
        logger.info(f"It worked! Here are your maps {os.path.abspath(wdir)}!")
    else:
        logger.info(f"Directory {work2_dir} not found")


def get_go(wdir, path_to_pdbs):
    check_browsers()
    driver = init_webdriver(wdir)
    get_go_maps(driver, path_to_pdbs)
    extract_go_maps(wdir)


if __name__ == "__main__":
    args = parse_arguments()
    WDIR = args.directory
    PDB_FILE = args.file
    DOWNLOADS_ABSPATH = os.path.abspath(WDIR)

    check_browsers()
    driver = init_webdriver()
    get_go_maps(driver)
    extract_go_maps()
