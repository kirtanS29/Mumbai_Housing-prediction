import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import Part, UserContent, ModelContent

# =========================
# Page config
# =========================
st.set_page_config(page_title="Mayanagri Mumbai House Price Predictor", layout="wide")

# =========================
# Load model & columns
# =========================
# pipeline = joblib.load("model.joblib")
# model_columns = joblib.load("model_columns.pkl")  # columns used in training
import os
import joblib
import streamlit as st
from train_model import build_and_train_model

# =========================
# Load or Train Model
# =========================


MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    from train_model import voting_reg
    joblib.dump(voting_reg, MODEL_PATH)

pipeline = joblib.load(MODEL_PATH)

# =========================
# Load Gemini API key
# =========================
load_dotenv()
gemini_api_key = os.getenv("OPENAI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

# =========================
# Precomputed average price per sqft
# =========================
# (Your avg_price_per_sqft dictionary here, truncated for brevity)
avg_price_per_sqft = {'15th Road': 43750.0, '4 Bunglows': 21234.567901234568, 'Adaigaon': 14736.842105263158, 'Airoli': 12718.369160087836, 'Akurli Nagar': 25446.426339285714, 'Almeida Park': 30937.5, 'Ambarnath': 3816.5725645821767, 'Ambernath East': 9299.329228359053, 'Ambernath West': 15484.1721084591, 'Ambivali': 31963.470319634704, 'Ambivli': 7031.25, 'Anand Nagar Thane West': 10240.0, 'Andheri': 18035.404405252466, 'Andheri East': 20199.482012127548, 'Andheri West': 14500.080627399631, 'Anjurdive': 15060.605454545455, 'Antop Hill': 25112.96703296703, 'Asangaon': 3545.2961672473866, 'Badlapur': 19294.31491102915, 'Badlapur East': 7655.898819104512, 'Badlapur West': 7141.428297954655, 'Balkum': 7115.6116553629745, 'Bandra East': 14934.858590770118, 'Bandra Kurla Complex': 34992.090972499005, 'Bandra West': 17541.458390944117, 'Bangur Nagar': 9166.666666666666, 'Belapur': 6235.150157432783, 'Beturkar Pada': 18725.49019607843, 'Bhakti Park': 11327.433628318584, 'Bhandup West': 11801.853645724957, 'Bhayandar East': 11938.172829808169, 'Bhayandar West': 8823.529411764706, 'Bhayandarpada': 5364.806866952789, 'Bhiwandi': 7236.482066426456, 'Bhoiwada Kalyan': 12820.51282051282, 'Boisar': 7631.5580617747155, 'Borivali': 22747.680641873303, 'Borivali East': 15728.182359609764, 'Borivali West': 18652.786253939463, 'Byculla': 82461.53846153847, 'CBD Belapur East': 12075.14880952381, 'Central Avenue': 8086.84831019125, 'Chandivali': 14761.904761904761, 'Charkop': 21712.703857495337, 'Charkop Sector 8': 16272.189349112426, 'Chedda Nagar': 3096.7741935483873, 'Chembur': 18492.49816473886, 'Chembur East': 18022.896659824197, 'Chembur Shell Colony': 23000.0, 'DN Nagar Road': 22666.666666666668, 'Dadar West': 8852.450179593941, 'Dahisar': 17518.042255087097, 'Dahisar East': 16978.07387207472, 'Dahisar W': 12307.692307692309, 'Dahisar West': 18980.416312767633, 'Dattapada': 16007.619529130603, 'Deonar': 15398.61494024233, 'Devidas Cross Lane': 17374.99875, 'Dharavi': 6461.538461538462, 'Diamond Market Road': 22733.333333333332, 'Diva': 13293.051359516616, 'Diva Gaon': 5043.478260869565, 'Dokali Pada': 13076.923076923076, 'Dombivali': 14153.775125988928, 'Dombivali East': 10330.075080699842, 'Dombivli (West)': 17983.762443964883, 'Dronagiri': 8622.84258252857, 'Eastern Express Highway Vikhroli': 13654.618473895582, 'Four Bungalows': 13366.013071895424, 'Four Bunglows': 28909.090909090908, 'Gandhar Nagar': 7040.845070422535, 'Ganesh Nagar': 28767.12328767123, 'Ghansoli': 15697.84395849499, 'Ghatkopar': 14181.442990284711, 'Ghatkopar East': 34481.19235103147, 'Ghatkopar West': 15330.430239066613, 'Ghodbunder Road': 9872.029250457039, 'Girgaon': 3345.3887884267633, 'Godrej Hill': 9693.475261186675, 'Goregaon': 16787.51253129817, 'Goregaon (East)': 4626.3345195729535, 'Goregaon East': 14608.858382447657, 'Goregaon West': 13437.578503146562, 'Govandi': 19360.890513374314, 'Govind nagar': 11466.666666666666, 'Grant Road West': 2133.333333333333, 'Greater Khanda': 19170.609450455406, 'Gulal Wadi': 3200.0, 'Gundavali Gaothan': 1967.2727272727273, 'Haridas Nagar': 23983.983983983984, 'Haware City': 6616.634784265174, 'Hiranandani Estates': 9439.36325131921, 'Hiranandani Meadows': 19802.60405919287, 'I C Colony': 13750.0, 'IT Colony': 6767.428679387871, 'Jankalyan Nagar': 11394.845810177618, 'Jawahar Nagar': 33981.143334526794, 'Jeejamata Nagar': 8479.404031551272, 'Jogeshwari East': 18743.891183597585, 'Jogeshwari West': 16810.613135872874, 'Juhu': 10310.390563895844, 'KASHELI': 13759.01875901876, 'Kalamboli': 14062.374770778322, 'Kalpataru': 15918.367346938776, 'Kalwa': 11804.399816417133, 'Kalyan': 6156.500108154877, 'Kalyan East': 8909.36788335147, 'Kalyan West': 11362.80982329677, 'Kamothe': 10292.355988544636, 'Kamothe Sector 16': 6489.960247743057, 'Kandivali East': 16996.587836322404, 'Kandivali West': 16289.439472711107, 'Kanjurmarg': 19505.40451486529, 'Kanjurmarg East': 10272.014753342555, 'Kannamwar Nagar II': 16988.636363636364, 'Kapur Bawdi': 2053.5714285714284, 'Kapurbawadi': 8947.368421052632, 'Karanjade': 14704.779167955783, 'Karave Nagar': 3750.0, 'Karjat': 3166.349423101698, 'Kasar vadavali': 6353.350739773717, 'Kasheli': 4243.329719474757, 'Katrap': 5392.156862745098, 'Kewale': 9039.422206595273, 'Khalapur': 5015.9759288330715, 'Khar': 31120.43208047151, 'Khar West': 9455.160391654792, 'Kharegaon': 8058.528876031287, 'Kharghar': 9228.899665945219, 'Kharghar Sector 34C': 8655.737704918032, 'Kharodi': 62000.0, 'Khopoli': 4999.757709450496, 'Koldongri': 4363.636363636364, 'Kolshet Industrial Area': 9079.903147699757, 'Kolshet Road': 14268.912684836498, 'Kondivita Road': 31250.0, 'Kopar Khairane Sector 19A': 10284.09090909091, 'Kopara': 16225.448334756618, 'Koparkhairane Station Road': 9258.932649541242, 'Koper Khairane': 13409.176264741183, 'Koproli': 12611.12435888046, 'Krishanlal Marwah Marg': 23595.505617977527, 'Kulupwadi': 15263.157894736842, 'Kurla': 20225.86361108609, 'Kurla West': 13891.266450478388, 'Link Road': 18013.88888888889, 'Lokhandwala': 12365.591397849463, 'Lokhandwala Township': 11666.666666666666, 'Lower Parel': 8661.702423299244, 'MHADA Colony 20': 12214.25467188179, 'Magathane': 18675.620823281017, 'Maharashtra Nagar': 10655.737704918032, 'Mahatma Gandhi Road': 19200.0, 'Mahim': 19069.343065693432, 'Majiwada': 11806.985009690046, 'Majiwada thane': 11094.725646404413, 'Malad East': 16495.527199548873, 'Malad West': 15958.055463514573, 'Manpada': 5797.94787006419, 'Manvel pada Road': 15686.274509803921, 'Marol': 8416.480520702384, 'Matunga': 9992.50136674345, 'Mira Bhayandar': 8965.51724137931, 'Mira Road': 10464.014535551827, 'Mira Road East': 12940.813603443094, 'Mira Road and Beyond': 8530.911699025099, 'Mulund': 13651.767280110298, 'Mulund East': 20546.6105274591, 'Mulund West': 18512.709287826572, 'Mumbai Agra National Highway': 8382.682132682134, 'Mumbai Central': 6394.484250563835, 'Mumbai Highway': 10894.199171919154, 'Mumbai Nashik Expressway': 9273.413566739606, 'Nahur': 26122.338992185396, 'Naigaon East': 13509.593268762777, 'Nala Sopara': 7846.993736774517, 'Nalasopara East': 12774.426613000289, 'Nalasopara West': 5217.750512807949, 'Natakwala Lane': 17017.017017017017, 'Navi Basti': 8156.619866488621, 'Neral': 6060.536691845439, 'Nere': 38405.79710144927, 'Nerul': 11286.350978084045, 'Nilje Gaon': 6051.587301587301, 'Off Nepean Sea Road': 7833.333333333333, 'Off Shimpoli road': 22777.777777777777, 'Owale': 10251.703694326645, 'PARSIK NAGAR': 9880.446720040629, 'Padle Gaon': 7697.841726618705, 'Palava': 7068.404220578133, 'Palghar': 8105.776896574976, 'Pali Hill': 43750.0, 'Palidevad': 6000.0, 'Palm Beach': 14042.553191489362, 'Panch Pakhadi': 15536.49588053553, 'Pandurangwadi': 18000.0, 'Pant Nagar': 13110.545368609884, 'Panvel': 10571.107574876036, 'Parel': 14006.42055255019, 'Patel Nagar': 9230.76923076923, 'Patlipada': 18095.238095238095, 'Petali': 7465.332631209409, 'Pokharan Road': 17187.5, 'Pokhran 2': 13007.19696969697, 'Pokhran Road No 2': 17066.666666666668, 'Powai': 17729.85141824237, 'Powai Lake': 9040.153349475384, 'Prabhadevi': 7462.17008797654, 'Rajendra Nagar': 18998.235342772296, 'Ramdev Park': 19060.052219321147, 'Ranjanpada': 8920.18779342723, 'Rasayani': 7261.410788381742, 'Rawal Pada': 17585.55133079848, 'Roadpali': 10682.754215599853, 'Rustomjee Global City': 6888.888888888889, 'Rutu Enclave': 4657.534246575343, 'Sahkar Nagar': 13529.410784313726, 'Sainath Nagar': 14422.348484848486, 'Saki Naka': 6360.708534621578, 'Samata Nagar Thakur Village': 19351.72570390554, 'Samata nagar': 14662.545787545787, 'Sanpada': 16450.970991792376, 'Santacruz East': 21663.573155900267, 'Santacruz West': 26568.375747863247, 'Seawoods': 13157.733630403663, 'Sector 10': 8925.362136668684, 'Sector 10 Kamothe': 2896.551724137931, 'Sector 10 Khanda Colony': 9150.32679738562, 'Sector 11 Belapur': 9729.27241962775, 'Sector 11 Kamothe': 8301.88679245283, 'Sector 11 Kharghar': 7708.653353814644, 'Sector 12 A': 10476.190476190477, 'Sector 12 Kharghar': 10431.438127090301, 'Sector 15': 10718.351033458946, 'Sector 15 Kharghar': 11008.966286239014, 'Sector 17 Ulwe': 9575.828774477537, 'Sector 18': 10428.60107928601, 'Sector 18 Kamothe': 9788.591549295776, 'Sector 18 Kharghar': 10182.955004389161, 'Sector 19 Kamothe': 7500.0, 'Sector 19 Kharghar': 9052.845908514004, 'Sector 19 Nerul': 10200.0, 'Sector 19A Nerul': 26666.666666666668, 'Sector 2 Ulwe': 26540.284360189573, 'Sector 20 Kamothe': 7542.4067795168385, 'Sector 20 Kharghar': 9336.119175607704, 'Sector 21 Kamothe': 7861.775347375346, 'Sector 21 Kharghar': 6923.809523809524, 'Sector 21 Ulwe': 7563.0252100840335, 'Sector 22 Kamothe': 10048.526422764227, 'Sector 23 Ulwe': 6271.153846153846, 'Sector 30': 8461.538461538461, 'Sector 30 Kharghar': 6237.562189054726, 'Sector 35G': 9343.814826946174, 'Sector 35I Kharghar': 9660.887622488359, 'Sector 36 Kamothe': 8000.0, 'Sector 36 Kharghar': 10066.854990583804, 'Sector 5': 9242.957746478873, 'Sector 5 Ulwe': 7642.29666674643, 'Sector 58A Seawoods Navi Mumbai': 3000.0, 'Sector 6': 5678.233438485804, 'Sector 7 Kharghar': 10736.196319018405, 'Sector 9 Airoli': 8018.327605956472, 'Sector-12 Kamothe': 9927.9176201373, 'Sector-13 Kharghar': 7664.233576642336, 'Sector-18 Ulwe': 9172.677561266086, 'Sector-24 Kamothe': 22419.35322580645, 'Sector-26 Taloja': 4666.666666666667, 'Sector-3 Ulwe': 8497.94128070448, 'Sector-34B Kharghar': 9658.536585365853, 'Sector-35 Kamothe': 9142.857142857143, 'Sector-5 Kamothe': 6785.714285714285, 'Sector-50 Seawoods': 9696.969696969696, 'Sector-6A Kamothe': 6206.896551724138, 'Sector-8 Sanpada': 19345.238095238095, 'Sector-8 Ulwe': 8437.231933882498, 'Sector-9 Ulwe': 27753.30396475771, 'Sector12 Kamothe': 11306.984225760523, 'Sector12 New Panvel': 7142.857142857143, 'Sector13 Khanda Colony': 10000.0, 'Sector13 Kharghar': 6727.272727272727, 'Sector16 Airoli': 9101.396478445658, 'Sector16 Ulwe': 3538.4615384615386, 'Sector34 A Kharghar': 7692.307692307692, 'Sector35D Kharghar': 7567.222222222222, 'Sector8 Sanpada': 57640.75067024129, 'Sector9 Kamothe': 9310.185591010866, 'Sen Nagar': 14920.63492063492, 'Seven Bunglow': 3520.3520352035202, 'Sewri': 25819.67213114754, 'Shakti Nagar': 10678.321678321678, 'Shastri Nagar': 27619.04761904762, 'Shil Phata': 23533.083292961604, 'Shilphata Road Thane': 4858.757062146893, 'Shimpoli': 23333.333333333332, 'Shirgaon': 6233.559958289885, 'Shivaji Colony': 37704.91803278688, 'Shreyas Colony': 13636.363636363636, 'Sindhi Society Chembur': 17432.052483598876, 'Sion': 12089.992539152969, 'Soniwadi Road': 23333.333333333332, 'Sriprastha': 4615.079365079366, 'Suburbs Mumbai': 14545.454545454546, 'Suman Nagar': 14752.475247524753, 'Sunil Nagar': 8333.333333333334, 'Syndicate': 5974.025974025974, 'TPS Road': 15416.666666666666, 'Taloja': 7355.729051409461, 'Taloja Bypass Nitalas Link Road': 5072.463768115942, 'Taloje': 7407.407407407408, 'Thakur Village': 16902.61813783543, 'Thakur complex': 18595.189353971557, 'Thakurli': 8400.0, 'Thane': 10691.224694075294, 'Thane Belapur Road Kalwa': 13568.78306878307, 'Thane West': 15092.09110619826, 'Tilak Nagar': 16086.95652173913, 'Tilak Nagar Mumbai': 5769.2307692307695, 'Titwala': 11654.793564041098, 'Tolaram Colony': 17083.333333333336, 'ULWE SECTOR 19': 5692.307692307692, 'Ulwe': 9554.79442860834, 'Uran': 14453.561253561254, 'Vakola': 24111.11111111111, 'Vakola Pipeline Road': 11000.0, 'Vangani': 9814.09037630361, 'Vartak Nagar': 12649.842693050283, 'Vasai': 12040.356265537193, 'Vasai West': 9626.200614024043, 'Vasai east': 15580.533467945674, 'Vasant Vihar': 24617.407839792897, 'Vashi': 6087.270380612642, 'Vasind': 3703.703703703704, 'Vazira': 10864.197530864198, 'Vedant Complex': 3225.0, 'Versova': 3297.872340425532, 'Vichumbe': 24444.444444444445, 'Vijay Nagar': 23023.25581395349, 'Vikhroli': 17481.898543694813, 'Vikhroli West': 21463.67521367521, 'Vikroli East': 19030.303030303032, 'Vile Parle': 31578.947368421053, 'Ville Parle East': 23580.738724240877, 'Ville Parle West': 12971.470892323076, 'Virar': 8590.961671671655, 'Virar East': 8570.501297308372, 'Virar West': 8129.825341740586, 'Vitthalwadi': 1500.0, 'Vivek Vidyalaya Marg': 20000.0, 'Wadala': 20339.62747878277, 'Wadala East': 11319.15494399191, 'Wadala East Wadala': 19758.141997743907, 'Wadi Bandar': 17684.21052631579, 'West Amardeep Colony': 16000.0, 'Western Express Highway Kandivali East': 18181.81818181818, 'Willingdon': 13205.282112845138, 'Worli': 12889.440230337323, 'Yogi Hills': 5294.117647058823, 'azad nagar': 33870.967741935485, 'dhanukarwadi': 8542.713567839195, 'gokuldham': 26000.0, 'kandivali': 16558.58523210539, 'kavesar': 6716.97579374963, 'kolshet': 13841.261808367073, 'link road borivali west': 5769.2307692307695, 'matunga east': 18213.674190102152, 'mumbai': 12141.902561597642, 'no 9': 19259.25925925926, 'raheja vihar': 18351.709401709402, 'roadpali navimumbai': 5947.859545090363, 'royal palms goregaon east': 14311.926605504586, 'sec 50 new': 19000.0, 'taloja panchanand': 7066.737846504772, 'thakur village kandivali east': 17458.79120879121, 'ulhasnagar 4': 2050.0, 'vasant vihar thane west': 16787.344794390086, 'vile parle west': 8604.683612960242, 'vrindavan society': 10458.715596330276}
# Your locations list
locations = sorted(['Kharghar', 'Thane West', 'Mira Road East', 'Ulwe', 'Nala Sopara', 'Borivali West', 'Kalyan West', 'Andheri West', 'Panvel', 'Powai', 'Malad West', 'Chembur', 'Kandivali East', 'Virar', 'Kandivali West', 'Kamothe', 'Goregaon West', 'Andheri East', 'Malad East', 'Mulund West', 'Boisar', 'Dahisar', 'Taloja', 'Ville Parle East', 'Goregaon East', 'Magathane', 'Naigaon East', 'Dombivali', 'Thane', 'Vasai', 'Seawoods', 'Borivali East', 'Badlapur East', 'Bhandup West', 'Ghatkopar West', 'Juhu', 'Parel', 'Dronagiri', 'Sanpada', 'Sector 17 Ulwe', 'Kalwa', 'Sector 20 Kharghar', 'Ghansoli', 'Koproli', 'Dombivali East', 'Karanjade', 'Jogeshwari West', 'Airoli', 'Kanjurmarg', 'Belapur', 'Koper Khairane', 'Bhiwandi', 'Dahisar West', 'Nalasopara West', 'Mulund East', 'Titwala', 'Mira Road and Beyond', 'Wadala', 'Dattapada', 'Bandra West', 'Dombivli (West)', 'Worli', 'Santacruz East', 'Vasai West', 'Dahisar East', 'Kurla', 'matunga east', 'Ambernath West', 'Nerul', 'Karjat', 'Wadala East Wadala', 'Goregaon', 'Virar East', 'Vasai east', 'Bhayandar East', 'mumbai', 'Thakur Village', 'Virar West', 'Ambernath East', 'Chembur East', 'Nalasopara East', 'Bandra East', 'Andheri', 'Khar West', 'Mulund', 'Kalyan East', 'Santacruz West', 'Sector-18 Ulwe', 'Vashi', 'Sector12 Kamothe', 'Sion', 'Sector9 Kamothe', 'Vasant Vihar', 'Sector-3 Ulwe', 'Majiwada', 'Ghatkopar', 'Kolshet Road', 'Palghar', 'Lower Parel', 'Sector 19 Kharghar', 'Kurla West'])

# =========================
# Streamlit title
# =========================
st.title("🏡 Bombay Price Prediction App")
st.markdown("Enter input values to predict house prices and view Gemini explanations.")

# =========================
# Input Form
# =========================
with st.form("input_form"):
    Area = st.number_input("Area (sqft)", min_value=200, max_value=5000, value=720)
    Bedrooms = st.slider("No. of Bedrooms", 1, 10, 2)
    Location = st.selectbox("Location", locations)
    Resale = st.selectbox("Resale", [0, 1])

    # Amenities
    Gymnasium = st.selectbox("Gymnasium", [0, 1])
    SwimmingPool = st.selectbox("Swimming Pool", [0, 1])
    ClubHouse = st.selectbox("Club House", [0, 1])
    CarParking = st.selectbox("Car Parking", [0, 1])
    PowerBackup = st.selectbox("Power Backup", [0, 1])
    LiftAvailable = st.selectbox("Lift Available", [0, 1])
    VaastuCompliant = st.selectbox("Vaastu Compliant", [0, 1])
    Security247 = st.selectbox("24X7 Security", [0, 1])

    submitted = st.form_submit_button("Predict")

# =========================
# Prediction
# =========================
if submitted:
    Location_AvgPrice = avg_price_per_sqft.get(Location, 0)
    amenities = [Gymnasium, SwimmingPool, ClubHouse, CarParking, PowerBackup, LiftAvailable, VaastuCompliant, Security247]
    Amenity_Count = sum(amenities)

    user_input = {
        "Area": Area,
        "No. of Bedrooms": Bedrooms,
        "Resale": Resale,
        "Location_AvgPrice": Location_AvgPrice,
        "Amenity_Count": Amenity_Count
    }

    input_df = pd.DataFrame([user_input])
    input_df = input_df[model_columns]

    # =========================
    # Model Prediction
    # =========================
    prediction = pipeline.predict(input_df)[0]
    if input_df["Resale"].iloc[0] == 1:
        prediction *= 0.850

    st.success(f"🏠 Model Predicted Price: ₹{prediction:,.0f}")

    # =========================
    # Gemini Prediction
    # =========================
    prompt = f"""
    You are a real estate expert in Mumbai. 
    Predict the price of a house based on the following details:
    Area: {Area} sqft
    Bedrooms: {Bedrooms}
    Location: {Location}
    Resale: {'Yes' if Resale==1 else 'No'}
    Amenities count: {Amenity_Count} (Gym, Pool, Clubhouse, Parking, PowerBackup, Lift, Vaastu, Security)
    
    Provide only the estimated price in INR without extra explanation.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        gpt_reply = response['choices'][0]['message']['content'].strip()
        st.info(f"💡 Gemini Prediction & Explanation:\n{gpt_reply}")
        
    except Exception as e:
        st.error(f"Error contacting Gemini API: {e}")
