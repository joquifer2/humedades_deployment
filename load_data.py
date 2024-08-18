# Importa PyDrive y las bibliotecas asociadas.

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import gspread
from google.auth import default

# Autentica y crea el cliente PyDrive.

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Autentica con gspread

creds, _ = default()
gc = gspread.authorize(creds)

# Reemplaza con el ID correcto del documento de Google Sheets
file_id = '1VoU01nkeAXRk3h6Eys2MPbHZXVx37smhJfY9x49OVAU'

# Abre el documento de Google Sheets
sheet = gc.open_by_key(file_id)

# Selecciona la primera hoja de cálculo (índice 0)
worksheet = sheet.get_worksheet(0)

# Obtén todos los valores de la hoja de cálculo
data = worksheet.get_all_values()