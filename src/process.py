import pandas as pd
import numpy as np
import re
import ast
from datetime import datetime

def determine_threshold(df, column, top_percent=95):
    """
    Определяет порог значимости для категориальных признаков, оставляя значения,
    которые покрывают указанный процент от всех данных.

    Параметры:
    - df: DataFrame с данными.
    - column: Название столбца с категориальными значениями.
    - top_percent: Процент данных, которые должны быть покрыты значимыми значениями (по умолчанию 95%).

    Возвращает:
    - Пороговое значение частоты.
    """
    # Подсчет частоты значений
    value_counts = df[column].value_counts()

    # Вычисление кумулятивной доли (процентов)
    cumulative_percent = value_counts.cumsum() / value_counts.sum() * 100

    # Определение порогового значения, чтобы покрыть top_percent процентов данных
    threshold_value = value_counts[cumulative_percent <= top_percent].min()

    return threshold_value

def reduce_categories(df, column, threshold):
    """
    Сокращает количество категорий, заменяя редкие значения на 'other'.

    Параметры:
    - df: DataFrame с данными.
    - column: Название столбца с категориальными значениями.
    - threshold: Минимальное количество появлений, чтобы оставить категорию значимой.

    Возвращает:
    - DataFrame с преобразованным столбцом.
    """
    # Подсчет частот значений
    value_counts = df[column].value_counts()

    # Определение значимых значений (те, что встречаются чаще порога)
    significant_values = value_counts[value_counts >= threshold].index

    # Замена всех редких значений на 'other'
    df[column] = df[column].apply(lambda x: x if x in significant_values else 'other')

    return df

def process_status(df):
    print("Обработка столбца: status", flush=True)
    df['status'] = df['status'].apply(lambda x: str(x).lower())
    def clean_status(value):
        if 'for sale' in value: return 'for sale'
        elif 'active' in value: return 'active'
        elif 'foreclosure' in value: return 'foreclosure'
        elif 'new' in value: return 'new'
        elif 'pending' in value: return 'pending'
        elif 'auction' in value: return 'auction'
        elif 'nan' in value or pd.isna(value): return 'no info'
        else: return 'other'

    df['status'] = df['status'].apply(lambda x: clean_status(x))
    return df


def process_private_pool(df):
    print("Обработка столбца: private pool / PrivatePool", flush=True)
    df['private_pool'] = df['private pool'].combine_first(df['PrivatePool'])
    df.drop(['private pool', 'PrivatePool'], axis=1, inplace=True)
    df['private_pool'] = df['private_pool'].apply(lambda x: str(x).lower())
    df['private_pool'] = df['private_pool'].replace('yes', 1).replace('nan', 0).fillna(0)
    return df


def process_property_type(df):
    print("Обработка столбца: propertyType", flush=True)
    df['propertyType'] = df['propertyType'].apply(lambda x: str(x).lower())

    def clean_propertyType(value):
        if 'nan' in value or pd.isna(value): return 'no info'
        elif 'single-family' in value or 'single family' in value: return 'single-family'
        elif 'multi-family' in value or 'multi family' in value: return 'multi-family'
        elif 'coop' in value or 'cooperative' in value: return 'cooperative'
        elif 'semi-detached' in value: return 'semi-detached'
        elif 'detached' in value: return 'detached'
        elif 'ranch' in value: return 'ranch'
        elif 'condo' in value: return 'condo'
        elif 'lot/land' in value or 'lot' == value or 'land' == value: return 'lot/land'
        elif 'townhouse' in value or 'townhome' in value: return 'townhouse'
        elif 'high rise' in value: return 'high rise'
        elif 'mobile/manufactured' in value or 'mobile' in value or 'manufactured' in value: return 'mobile/manufactured'
        elif 'apartment' in value: return 'apartment'
        else: return 'other'

    df['propertyType'] = df['propertyType'].apply(lambda x: clean_propertyType(x))
    return df


def process_target(df):
    print("Обработка столбца: target", flush=True)
    def process_target(row):
        try:
            value = row['target']
            isRental = 0

            if pd.isnull(value):
                return pd.Series([None, None])

            value_cleaned = value.replace(',', '').replace('$', '').replace('+', '')

            if '/mo' in value: # Съемное жилье
                value_cleaned = value_cleaned.replace('/mo', '')
                isRental = 1

            if '-' in value_cleaned:
                low, high = map(int, value_cleaned.split('-'))
                value_cleaned = (low + high) // 2  # Среднее значение диапазона

            return pd.Series([int(value_cleaned), isRental])  # 1 - арендная плата
        except:
            print(value)
    df[['target_cleaned', 'is_rental']] = df.apply(process_target, axis=1)
    return df


def process_street(df):
    print("Обработка столбца: street", flush=True)
    # Полный справочник типов улиц
    street_types = {
        'alley': ['allee', 'ally', 'aly'],
        'annex': ['anex', 'annx', 'anx'],
        'arcade': ['arc'],
        'avenida': [],
        'avenue': ['av', 'ave', 'aven', 'avenu', 'avn', 'avnue'],
        'bayou': ['bayoo', 'byu'],
        'beach': ['bch'],
        'bend': ['bnd'],
        'bluff': ['bluf', 'blf'],
        'bluffs': ['blfs'],
        'bottom': ['bot', 'bottm', 'btm'],
        'boulevard': ['boul', 'boulv', 'bld', 'blvd'],
        'branch': ['brnch', 'br'],
        'bridge': ['brdge', 'brg'],
        'brook': ['brk'],
        'brooks': ['brks'],
        'burg': ['bg'],
        'burgs': ['bgs'],
        'bypass': ['bypa', 'bypas', 'byps', 'byp'],
        'calle': [],
        'camino': [],
        'camp': ['cmp', 'cp'],
        'canyon': ['canyn', 'cnyn', 'cyn'],
        'cape': ['cpe'],
        'causeway': ['causwa', 'cswy'],
        'center': ['cen', 'cent', 'centr', 'centre', 'cnter', 'cntr', 'ctr'],
        'centers': ['ctrs'],
        'circle': ['circ', 'circl', 'crcl', 'crcle', 'cir'],
        'circles': ['cirs'],
        'cliff': ['clf'],
        'cliffs': ['clfs'],
        'club': ['clb'],
        'common': ['cmn'],
        'commons': ['cmns'],
        'corner': ['cor'],
        'corners': ['cors'],
        'course': ['crse'],
        'court': ['ct'],
        'courts': ['cts'],
        'cove': ['cv'],
        'coves': ['cvs'],
        'creek': ['crk'],
        'crescent': ['crsent', 'crsnt', 'cres'],
        'crest': ['crst'],
        'crossing': ['crssng', 'xing'],
        'crossroad': ['xrd'],
        'curve': ['curv'],
        'dale': ['dl'],
        'dam': ['dm'],
        'divide': ['div', 'dvd', 'dv'],
        'drive': ['driv', 'drv', 'dr'],
        'drives': ['drs'],
        'estate': ['est'],
        'estates': ['ests'],
        'expressway': ['exp', 'expr', 'express', 'expw', 'expwy', 'expy'],
        'extension': ['extn', 'extnsn', 'ext'],
        'extensions': ['exts'],
        'fall': [],
        'falls': ['fls'],
        'ferry': ['frry', 'fry'],
        'field': ['fld'],
        'fields': ['flds'],
        'flat': ['flt'],
        'flats': ['flts'],
        'ford': ['frd'],
        'fords': ['frds'],
        'forest': ['frst'],
        'forge': ['forg', 'frg'],
        'forges': ['frgs'],
        'fork': ['frk'],
        'forks': ['frks'],
        'fort': ['frt', 'ft'],
        'freeway': ['freewy', 'frway', 'frwy', 'fwy'],
        'garden': ['gardn', 'grden', 'grdn', 'gdn'],
        'gardens': ['gdns'],
        'gateway': ['gatewy', 'gatway', 'gtway', 'gtwy'],
        'glen': ['gln'],
        'glens': ['glns'],
        'green': ['grn'],
        'greens': ['grns'],
        'grove': ['grov', 'grv'],
        'groves': ['grvs'],
        'harbor': ['harb', 'harbr', 'hrbor', 'hbr'],
        'harbors': ['hbrs'],
        'haven': ['hvn'],
        'heights': ['hts'],
        'highway': ['highwy', 'hiway', 'hiwy', 'hway', 'hwy'],
        'hill': ['hl'],
        'hills': ['hls'],
        'hollow': ['hllw', 'holw', 'holws'],
        'inlet': ['inlt'],
        'island': ['is'],
        'islands': ['iss'],
        'isle': [],
        'junction': ['jction', 'jctn', 'junctn', 'juncton', 'jct'],
        'junctions': ['jcts'],
        'key': ['ky'],
        'keys': ['kys'],
        'knoll': ['knol', 'knl'],
        'knolls': ['knls'],
        'lake': ['lk'],
        'lakes': ['lks'],
        'land': [],
        'landing': ['lndng', 'lndg'],
        'lane': ['la', 'ln'],
        'light': ['lgt'],
        'lights': ['lgts'],
        'loaf': ['lf'],
        'lock': ['lck'],
        'locks': ['lcks'],
        'lodge': ['ldge', 'lodg', 'ldg'],
        'loop': ['lp'],
        'mall': [],
        'manor': ['mnr'],
        'manors': ['mnrs'],
        'meadow': ['mdw'],
        'meadows': ['medows', 'mdws'],
        'mews': [],
        'mill': ['ml'],
        'mills': ['mls'],
        'mission': ['msn'],
        'motorway': ['mtwy'],
        'mount': ['mt'],
        'mountain': ['mtn'],
        'mountains': ['mtns'],
        'neck': ['nck'],
        'orchard': ['orchrd', 'orch'],
        'oval': ['ovl'],
        'overlook': ['ovlk'],
        'overpass': ['opas'],
        'park': ['prk'],
        'parks': ['park'],
        'parkway': ['parkwy', 'pkway', 'pky', 'pkwy'],
        'parkways': ['pkwys'],
        'pass': [],
        'passage': ['psge'],
        'path': [],
        'pike': ['pk'],
        'pine': ['pne'],
        'pines': ['pnes'],
        'place': ['pl'],
        'plain': ['pln'],
        'plains': ['plns'],
        'plaza': ['plza', 'plz'],
        'point': ['pt'],
        'points': ['pts'],
        'port': ['prt'],
        'ports': ['prts'],
        'prairie': ['prr', 'pr'],
        'radial': ['rad', 'radiel', 'radl'],
        'ramp': ['rmp'],
        'ranch': ['rnch', 'rnchs'],
        'rapid': ['rpd'],
        'rapids': ['rpds'],
        'rest': ['rst'],
        'ridge': ['rdge', 'rdg'],
        'ridges': ['rdgs'],
        'river': ['rvr', 'rivr', 'riv'],
        'road': ['rd'],
        'roads': ['rds'],
        'route': ['rte'],
        'row': [],
        'rue': [],
        'run': [],
        'shoal': ['shl'],
        'shoals': ['shls'],
        'shore': ['shr'],
        'shores': ['shrs'],
        'skyway': ['skwy'],
        'spring': ['spng', 'sprng', 'spg'],
        'springs': ['spgs'],
        'spur': [],
        'square': ['sqr', 'sqre', 'squ', 'sq'],
        'squares': ['sqs'],
        'station': ['statn', 'stn', 'sta'],
        'strasse': [],
        'stravenue': ['strav', 'straven', 'stravn', 'strvn', 'strvnue', 'stra'],
        'stream': ['streme', 'strm'],
        'street': ['str', 'strt', 'st'],
        'streets': ['sts'],
        'summit': ['sumit', 'sumitt', 'smt'],
        'terrace': ['terr', 'ter'],
        'throughway': ['trwy'],
        'trace': ['trce'],
        'track': ['trak', 'trk', 'trks'],
        'trafficway': ['trfy'],
        'trail': ['trl'],
        'trailer': ['trlr'],
        'tunnel': ['tunl'],
        'turnpike': ['trnpk', 'turnpk', 'tpke'],
        'underpass': ['upas'],
        'union': ['un'],
        'unions': ['uns'],
        'valley': ['vally', 'vlly', 'vly'],
        'valleys': ['vlys'],
        'via': [],
        'viaduct': ['vdct', 'viadct', 'via'],
        'view': ['vw'],
        'views': ['vws'],
        'village': ['vill', 'villag', 'villg', 'vlg'],
        'villages': ['vlgs'],
        'ville': ['vl'],
        'vista': ['vist', 'vst', 'vsta', 'vis'],
        'walk': [],
        'wall': [],
        'way': ['wy'],
        'well': ['wl'],
        'wells': ['wls']
    }

    street_types_pattern = r'\b(?:' + '|'.join([re.escape(item) for sublist in [[k] + v for k, v in street_types.items()] for item in sublist]) + r')\b'

    def extract_street_type(address):
        address = str(address).lower()
        match = re.search(street_types_pattern, address, re.IGNORECASE)
        if match:
            for street_type, variations in street_types.items():
                if match.group(0) in [street_type] + variations:
                    return street_type
        return 'no info'

    df['street_type'] = df['street'].apply(lambda x: extract_street_type(x))
    threshold = determine_threshold(df, 'street_type', top_percent=95)
    df = reduce_categories(df, 'street_type', threshold=threshold)
    return df


def process_baths(df):
    print("Обработка столбца: baths", flush=True)
    def convert_baths_to_numeric(value):
        if pd.isna(value):
            return 0
        value = str(value).lower()
        value = value.replace('baths', '').replace('bathrooms:', '').replace('ba', '').replace(',', '.').replace('+', '').strip()

        # Обрабатываем сложные случаи, например, '1 / 1-0 / 1-0 / 1-0'
        if '/' in value or '-' in value:
            return 1

        try:
            return float(value)
        except ValueError:
            return 0

    df['baths_numeric'] = df['baths'].apply(convert_baths_to_numeric)

    # Функция для категоризации
    def categorize_baths(value):
        if value == 0:
            return '0'
        elif value == 1:
            return '1'
        elif value == 2:
            return '2'
        elif value == 3:
            return '3'
        elif value == 4:
            return '4'
        else:
            return '5+'

    # Применение функции к столбцу
    df['baths_category'] = df['baths_numeric'].apply(categorize_baths)
    return df

def process_homeFacts(df):
    print("Обработка столбца: homeFacts", flush=True)
    def parse_home_facts(row):
        home_facts = ast.literal_eval(row['homeFacts'])
        facts = home_facts['atAGlanceFacts']
        result = {}
        for fact in facts:
            result[fact['factLabel']] = fact['factValue']
        return pd.Series(result)

    df_parsed = df.apply(parse_home_facts, axis=1)

    df = pd.concat([df, df_parsed], axis=1)

    df.drop('homeFacts', axis=1, inplace=True)

    # Функция для объединения по возрасту
    def categorize_by_age(year):
        try:
            year = int(year)
            current_year = datetime.now().year
            age = current_year - year

            # Классификация по возрасту
            if age <= 10:
                return "0-10"
            elif age <= 20:
                return "11-20"
            elif age <= 30:
                return "21-30"
            elif age <= 40:
                return "31-40"
            elif age <= 50:
                return "41-50"
            else:
                return "50+"
        except (ValueError, TypeError):
            return "no info"

    # Функция для проверки, прошло ли менее 10 лет с момента ремонта
    def remodeled_recently(year):
        if year and year.isdigit():
            return 1 if (datetime.now().year - int(year)) <= 10 else 0
        else:
            return 0

    df['Year built'] = df['Year built'].apply(categorize_by_age)
    df['Remodeled year'] = df['Remodeled year'].apply(remodeled_recently)

    def clean_field(value):
        if pd.isna(value):
            return int(0)
        value = str(value).lower()
        if 'no data' == value or 'none' == value or 'no' == value: return int(0)
        else: return int(1)

    df['Heating'] = df['Heating'].apply(lambda x: clean_field(x))
    df['Cooling'] = df['Cooling'].apply(lambda x: clean_field(x))
    df['Parking'] = df['Parking'].apply(lambda x: clean_field(x))

    def convert_lotsize_to_numeric(value):
        if pd.isna(value) or value.lower() in ['no data', '—', '--']:
            return 0
        value = str(value).lower().replace('sqft', '').replace('sq. ft.', '').replace('lot', '').replace(',', '').strip()

        # Если указаны акры, конвертируем в квадратные футы (1 acre = 43560 sqft)
        if 'acre' in value:
            try:
                acres = float(value.replace('acres', '').replace('acre', '').strip())
                return acres * 43560
            except ValueError:
                return 0
        else:
            try:
                return float(value)
            except ValueError:
                return 0

    df['lotsize_numeric'] = df['lotsize'].apply(convert_lotsize_to_numeric)

    # Находим моду и заменяем нулевые значения на моду
    mode_lotsize = df['lotsize_numeric'][df['lotsize_numeric'] != 0].mode()[0]
    df['lotsize_numeric'].replace(0, mode_lotsize, inplace=True)

    def convert_price_sqft_to_numeric(value):
        value = str(value).lower()
        value = value.replace('$', '').replace(',', '').replace('/sqft', '').replace(' / sq. ft.', '')
        try:
            return int(value)
        except:
            return 0

    df['Price/sqft_numeric'] = df['Price/sqft'].apply(convert_price_sqft_to_numeric)

    df.drop('Price/sqft', axis=1, inplace=True)
    return df


def process_fireplace(df):
    print("Обработка столбца: fireplace", flush=True)
    def clean_fireplace(value):
        if pd.isna(value):
            return int(0)
        value = str(value).lower()
        if 'no data' == value or 'none' == value or 'no' == value or '0' == value: return int(0)
        else: return int(1)

    df['fireplace'] = df['fireplace'].apply(lambda x: clean_fireplace(x))
    return df


def process_city(df):
    print("Обработка столбца: city", flush=True)
    df['city'].fillna('unknown', inplace=True)
    threshold = determine_threshold(df, 'city', top_percent=95)
    df = reduce_categories(df, 'city', threshold=threshold)
    return df


def process_schools(df):
    print("Обработка столбца: schools", flush=True)
    def process_school_info(schools):
        schools = ast.literal_eval(schools)

        ratings = []
        distances = []
        grades = []

        for school in schools:
            if isinstance(school, dict):
                if 'rating' in school and 'data' in school and 'Distance' in school['data']:
                    for rating, distance in zip(school['rating'], school['data']['Distance']):
                        # Исключаем 'NA' и 'NR'
                        if rating.isdigit() or '/' in rating:
                            if rating.split('/')[0].isdigit():
                                ratings.append(int(rating.split('/')[0]))
                                distances.append(float(distance.replace(' mi', '').replace('mi', '')))
                if 'data' in school and 'Grades' in school['data']:
                    grades.extend(school['data']['Grades'])

        # Взвешенный рейтинг
        if distances and ratings:
            weights = [1/(d if d != 0 else 0.1) for d in distances]
            weighted_rating = np.average(ratings, weights=weights)
        else:
            weighted_rating = np.nan

        # Бинарные признаки
        has_high_rated_school_nearby = int(any(r >= 8 for r in ratings)) if ratings else 0
        has_low_rated_school_nearby = int(any(r <= 4 for r in ratings)) if ratings else 0

        # Обработка классов обучения
        has_primary_school = int(any('PK' in g or 'K' in g or '5' in g or '4' in g for g in grades)) if grades else 0
        has_middle_school = int(any('6' in g or '8' in g for g in grades)) if grades else 0
        has_high_school = int(any('9' in g or '12' in g for g in grades)) if grades else 0

        return pd.Series({
            'Weighted_Rating': weighted_rating,
            'High_Rated_School_Nearby': has_high_rated_school_nearby,
            'Low_Rated_School_Nearby': has_low_rated_school_nearby,
            'Has_Primary_School': has_primary_school,
            'Has_Middle_School': has_middle_school,
            'Has_High_School': has_high_school
        })

    processed_data = df['schools'].apply(process_school_info)

    df = pd.concat([df, processed_data], axis=1)

    df['Weighted_Rating'].fillna(df['Weighted_Rating'].mean(), inplace=True)
    return df


def process_sqft(df):
    print("Обработка столбца: sqft", flush=True)
    def convert_sqft_to_numeric(row):
        value = row['sqft']
        if pd.isna(value):
            if row['Price/sqft_numeric'] != 0:
                return row['target_cleaned'] / row['Price/sqft_numeric']
            else:
                return 0
        else:
            value = str(value).lower()
            value = value.replace('sqft', '').replace(',', '').replace('total interior livable area: ','').replace('--', '0').replace('-', '').strip()
            return float(value)

    df['sqft_numeric'] = df.apply(convert_sqft_to_numeric, axis=1)
    df.drop('sqft', axis=1, inplace=True)
    return df


def process_beds(df):
    print("Обработка столбца: beds", flush=True)
    def convert_beds_to_numeric(value):
        if pd.isna(value):
            return 0
        value = str(value).lower()
        value = value.replace('beds', '')

        if 'sqft' in value or 'acres' in value:
            return 1

        try:
            return float(value)
        except ValueError:
            return 0

    df['beds_numeric'] = df['beds'].apply(convert_beds_to_numeric)

    def categorize_beds(value):
        if value == 0:
            return '0'
        elif value == 1:
            return '1'
        elif value == 2:
            return '2'
        elif value == 3:
            return '3'
        elif value == 4:
            return '4'
        else:
            return '5+'

    df['beds_category'] = df['beds_numeric'].apply(categorize_beds)
    return df


def process_state(df):
    print("Обработка столбца: state", flush=True)
    df['state'].fillna('unknown', inplace=True)
    threshold = determine_threshold(df, 'state', top_percent=90)
    df = reduce_categories(df, 'state', threshold=threshold)
    return df


def process_stories(df):
    print("Обработка столбца: stories", flush=True)
    def convert_to_number(value):
        value = str(value).lower()
        if value == 'nan': return 'no info'
        try:
            f = float(value)
            if f == 0:
                return '0'
            elif f == 1:
                return '1'
            elif f == 2:
                return '2'
            elif f == 3:
                return '3'
            else:
                return 'other'
        except ValueError:
            if '1' in value or 'one' in value: return '1'
            if '2' in value or 'two' in value: return '2'
            if '3' in value or 'three' in value or 'tri' in value: return '3'
            else: return 'other'

    df['stories_numeric'] = df['stories'].apply(convert_to_number)
    return df


def clean_data(df, train=True):
    df = process_status(df)
    df = process_private_pool(df)
    df = process_property_type(df)
    if train:
        df = process_target(df)
    df = process_street(df)
    df = process_baths(df)
    df = process_homeFacts(df)
    df = process_fireplace(df)
    df = process_city(df)
    df = process_schools(df)
    df = process_sqft(df)
    df = process_beds(df)
    df = process_state(df)
    df = process_stories(df)
    if train:
        df = df[df['is_rental'] == 0]
        df = df.drop(['is_rental'], axis=1)
    return df