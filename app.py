# import pandas as pd
# import folium
# from folium.plugins import HeatMap, Fullscreen, LocateControl
# from ipywidgets import interact, Dropdown, Checkbox, VBox, HBox, Label, Button
# from IPython.display import display, clear_output
# import datetime
# import h3
# from geojson import Feature, Point, FeatureCollection
# import json
# import matplotlib
# import scipy.stats as scs
# from ipywidgets import interact, IntSlider
# import IPython.display as display
# import warnings
# warnings.filterwarnings("ignore")
# from folium.plugins import HeatMap
# from sklearn.cluster import DBSCAN
# import numpy as np
# from folium.plugins import MarkerCluster
# import streamlit as st


# class SeasonalHeatmapVisualizer:
#     def __init__(self, data_path="metaData.csv"):
#         # Загрузка и подготовка данных
#         self.data = pd.read_csv(data_path, parse_dates=['photo_date'])
#         self._clean_data()
#         self._add_seasons()
        
#         # Настройки классов
#         self.class_config = {
#             1: {"name": "Памятники и артобъекты", "color": "red"},
#             2: {"name": "Архитектура", "color": "blue"},
#             3: {"name": "Торговые центры и магазины", "color": "purple"},
#             4: {"name": "Гастрономия", "color": "pink"},
#             5: {"name": "Реки, озера и фонтаны", "color": "lightblue"},
#             6: {"name": "Парки и зоны отдыха", "color": "darkgreen"}
#         }
        
#         # Создаем элементы управления
#         self._create_widgets()
        
#     def _clean_data(self):
#         """Очистка и валидация данных"""
#         initial_count = len(self.data)
#         self.data = self.data.dropna(subset=['photo_lat', 'photo_lon', 'predicted_class'])
        
#         # Фильтрация нереальных координат (пример для Казани)
#         self.data = self.data[
#             (self.data['photo_lat'].between(55.5, 56.0)) & 
#             (self.data['photo_lon'].between(48.5, 49.5))
#         ]
#         filtered_count = len(self.data)
        
#         if initial_count != filtered_count:
#             print(f"Отфильтровано {initial_count - filtered_count} записей с некорректными данными")

#     def _add_seasons(self):
#         """Добавляем информацию о сезонах"""
#         self.seasons = {
#             'Все сезоны': None,
#             'Зима': [12, 1, 2],
#             'Весна': [3, 4, 5],
#             'Лето': [6, 7, 8],
#             'Осень': [9, 10, 11]
#         }
        
#         self.data['season'] = self.data['photo_date'].apply(
#             lambda x: next(
#                 (season for season, months in self.seasons.items() 
#                  if months and x.month in months), 
#                 None
#             )
#         )

#     def _create_widgets(self):
#         """Создаем интерактивные элементы управления"""
#         self.season_dropdown = Dropdown(
#             options=list(self.seasons.keys()),
#             value='Все сезоны',
#             description='Сезон:',
#             style={'description_width': '50px'}
#         )
        
#         self.class_checkboxes = {
#             str(class_id): Checkbox(
#                 value=True,
#                 description=config['name'],
#                 indent=False
#             )
#             for class_id, config in self.class_config.items()
#         }
        
#         self.reset_button = Button(
#             description='Сбросить фильтры',
#             button_style='',
#             tooltip='Сбросить все фильтры'
#         )
#         self.reset_button.on_click(self._reset_filters)
        
#         self.info_label = Label()
#         self._update_info()
        
#         self.controls = VBox([
#             HBox([self.season_dropdown, self.reset_button]),
#             Label("Классы объектов:"),
#             VBox(list(self.class_checkboxes.values())),
#             self.info_label
#         ])

#     def _reset_filters(self, b):
#         """Сброс всех фильтров"""
#         self.season_dropdown.value = 'Все сезоны'
#         for checkbox in self.class_checkboxes.values():
#             checkbox.value = True
#         self._update_map()

#     def _update_info(self):
#         """Обновление информационной панели"""
#         total = len(self.data)
#         date_range = f"{self.data['photo_date'].min().date()} - {self.data['photo_date'].max().date()}"
#         self.info_label.value = f"Всего точек: {total} | Период: {date_range}"

#     def _filter_data(self):
#         """Фильтрация данных по выбранным параметрам"""
#         if self.season_dropdown.value != 'Все сезоны':
#             filtered = self.data[self.data['season'] == self.season_dropdown.value]
#         else:
#             filtered = self.data.copy()
        
#         selected_classes = [
#             int(class_id) for class_id, cb in self.class_checkboxes.items() 
#             if cb.value
#         ]
#         return filtered[filtered['predicted_class'].isin(selected_classes)]

#     def _create_map(self, filtered_data):
#         """Создание карты с отфильтрованными данными"""
#         m = folium.Map(
#             location=[filtered_data['photo_lat'].mean(), filtered_data['photo_lon'].mean()],
#             zoom_start=14,
#             tiles='cartodbpositron',
#             control_scale=True
#         )
#         Fullscreen().add_to(m)
#         LocateControl(auto_start=False).add_to(m)
        
#         for class_id, config in self.class_config.items():
#             class_data = filtered_data[filtered_data['predicted_class'] == class_id]
#             if not class_data.empty:
#                 HeatMap(
#                     class_data[['photo_lat', 'photo_lon']].values,
#                     radius=30,
#                     blur=15,
#                     name=config['name'],
#                     show=False,
                    
#                     min_opacity=0.4,
#                     max_zoom=14
#                 ).add_to(m)
        
        
#         folium.LayerControl(
#             position='topright',
#             collapsed=False,
#             autoZIndex=True
#         ).add_to(m)
        
#         return m

#     def _update_map(self, *args):
#         """Обновление карты"""
#         filtered_data = self._filter_data()
        
        
#         if not filtered_data.empty:
#             m = self._create_map(filtered_data)
#             display(m)
#         else:
#             print("Нет данных для отображения с текущими параметрами фильтрации")

#     def show(self):
#         """Запуск визуализации"""
        
#         self.season_dropdown.observe(self._update_map, 'value')
#         for checkbox in self.class_checkboxes.values():
#             checkbox.observe(self._update_map, 'value')
#         self._update_map()
#         display(self.controls)
# visualizer = SeasonalHeatmapVisualizer("metaData.csv")

# df = pd.read_csv("metaData.csv")
# df=df.rename(columns={"photo_id": "id", "photo_lat": "lat", "photo_lon":"long", 'photo_date': 'date'})
# print(df)
# df = df[['id', 'date', 'lat', 'long', 'predicted_class']].copy()
# df = df.rename(columns={
#     'lat': 'long',
#     'long': 'lat'
# })
# df= df.dropna(subset=['lat', 'long', 'predicted_class'])
# resolution = 9
# hex_ids = df.apply(lambda row: h3.latlng_to_cell(row['lat'], row['long'], resolution), axis=1)
# df = df.assign(hex_id=hex_ids.values)
# dfbyhexid = df.groupby("hex_id").size().reset_index(name='value')
# hex_ids.nunique()
# dfbyhexid["percentile"] = dfbyhexid["value"]
# def hexagons_dataframe_to_geojson(df_hex, file_output=None, column_name="value"):
#     list_features = []
    
#     for i,row in df_hex.iterrows():
#         try:
#             geometry_for_row = {"type": "Polygon", "coordinates": [h3.cell_to_boundary(h=row["hex_id"])]}
#             feature = Feature(geometry=geometry_for_row, id=row["hex_id"], properties={column_name: row[column_name]})
#             list_features.append(feature)
#         except:
#             print("An exception occurred for hex " + row["hex_id"]) 

#     feat_collection = FeatureCollection(list_features)
#     geojson_result = json.dumps(feat_collection)
#     return geojson_result

# def get_color(custom_cm, val, vmin, vmax):
#     return matplotlib.colors.to_hex(custom_cm((val-vmin)/(vmax-vmin)))

# def choropleth_map(df_aggreg, column_name="value", border_color='black', 
#                   fill_opacity=0.7, color_map_name="winter", initial_map=None):
#     min_value = df_aggreg[column_name].min()
#     max_value = df_aggreg[column_name].max()
#     mean_value = df_aggreg[column_name].mean()
#     print(f"Colour column min value {min_value}, max value {max_value}, mean value {mean_value}")
#     print(f"Hexagon cell count: {df_aggreg['hex_id'].nunique()}")
    
#     name_layer = "Choropleth " + str(df_aggreg)
    
#     if initial_map is None:
#         initial_map = folium.Map(location=[55.79, 49.12], zoom_start=11, tiles="cartodbpositron")

#     geojson_data = hexagons_dataframe_to_geojson(df_hex=df_aggreg, column_name=column_name)
#     custom_cm = matplotlib.cm.get_cmap(color_map_name)

#     folium.GeoJson(
#         geojson_data,
#         style_function=lambda feature: {
#             'fillColor': get_color(custom_cm, feature['properties'][column_name], vmin=min_value, vmax=max_value),
#             'color': border_color,
#             'weight': 1,
#             'fillOpacity': fill_opacity 
#         }, 
#         name=name_layer
#     ).add_to(initial_map)

#     return initial_map
# def interactive_h3_map(resolution=9):
#     hex_ids = df.apply(lambda row: h3.latlng_to_cell(row['lat'], row['long'], resolution), axis=1)
#     df_temp = df.assign(hex_id=hex_ids.values)
    
#     dfbyhexid_temp = df_temp.groupby("hex_id").size().reset_index(name='value')
    
#     map_ = choropleth_map(df_aggreg=dfbyhexid_temp, color_map_name="hot", column_name="value")
#     display.display(map_)

# interact(interactive_h3_map, resolution=IntSlider(min=6, max=15, step=1, value=9))
# hexmap = choropleth_map(df_aggreg=dfbyhexid, color_map_name="hot", column_name="value")

# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# def get_season(date):
#     if date.month in [12, 1, 2]:
#         return 'Зима'
#     elif date.month in [3, 4, 5]:
#         return 'Весна'
#     elif date.month in [6, 7, 8]:
#         return 'Лето'
#     else:
#         return 'Осень'
# df['season'] = df['date'].apply(get_season)
# resolution = 9
# hex_ids = df.apply(lambda row: h3.latlng_to_cell(row.lat, row.long, resolution), axis=1)
# df = df.assign(hex_id=hex_ids.values)


# def hexagons_dataframe_to_geojson(df_hex, column_name="value"):
#     list_features = []
#     for i, row in df_hex.iterrows():
#         try:
#             geometry_for_row = {"type": "Polygon", "coordinates": [h3.cell_to_boundary(h=row["hex_id"])]}
#             feature = Feature(geometry=geometry_for_row, id=row["hex_id"], properties={column_name: row[column_name]})
#             list_features.append(feature)
#         except Exception as e:
#             print(f"An exception occurred for hex {row['hex_id']}: {e}")
#     feat_collection = FeatureCollection(list_features)
#     geojson_result = json.dumps(feat_collection)
#     return geojson_result


# def get_color(custom_cm, val, vmin, vmax):
#     return matplotlib.colors.to_hex(custom_cm((val - vmin) / (vmax - vmin)))


# def choropleth_map(df_aggreg, column_name="value", border_color='black', fill_opacity=0.7, color_map_name="cool", initial_map=None, layer_name="Layer"):
#     min_value = df_aggreg[column_name].min()
#     max_value = df_aggreg[column_name].max()
#     mean_value = df_aggreg[column_name].mean()
#     print(f"Colour column min value {min_value}, max value {max_value}, mean value {mean_value}")
#     print(f"Hexagon cell count: {df_aggreg['hex_id'].nunique()}")

#     if initial_map is None:
#         initial_map = folium.Map(location=[55.79, 49.12], zoom_start=11)

#     geojson_data = hexagons_dataframe_to_geojson(df_hex=df_aggreg, column_name=column_name)
#     custom_cm = matplotlib.cm.get_cmap(color_map_name)

#     folium.GeoJson(
#         geojson_data,
#         style_function=lambda feature: {
#             'fillColor': get_color(custom_cm, feature['properties'][column_name], vmin=min_value, vmax=max_value),
#             'color': border_color,
#             'weight': 1,
#             'fillOpacity': fill_opacity
#         },
#         name=layer_name
#     ).add_to(initial_map)

#     return initial_map

# def update_map(resolution=9):
#     m = folium.Map(location=[55.79, 49.12], zoom_start=12)
#     hex_ids = df.apply(lambda row: h3.latlng_to_cell(row.lat, row.long, resolution), axis=1)
#     df_with_resolution = df.assign(hex_id=hex_ids.values)
#     seasons = ['Зима', 'Весна', 'Лето', 'Осень']
#     for season in seasons:
#         df_season = df_with_resolution[df_with_resolution['season'] == season]
#         if not df_season.empty:
#             dfbyhexid = df_season.groupby("hex_id").size().reset_index(name='value')
#             dfbyhexid["percentile"] = dfbyhexid["value"]
#             choropleth_map(
#                 df_aggreg=dfbyhexid, 
#                 color_map_name="hot", 
#                 column_name="value", 
#                 initial_map=m, 
#                 layer_name=f"{season}"
#             )
#     folium.LayerControl().add_to(m)
#     return m


# interact(update_map, resolution=IntSlider(min=6, max=15, step=1, value=9, description='Разрешение H3:'))

# kazan_meta = pd.read_csv('kazan_photos_metadata.csv')
# predicted_meta = pd.read_csv('metaData.csv')

# merged_data = pd.merge(kazan_meta, predicted_meta, on='photo_id', how='inner')
# merged_data = merged_data.head(1000)
# grouped = merged_data.groupby(['osm_object_name', 'osm_lat', 'osm_lon'])
# def create_object_heatmap(object_name, object_lat, object_lon, photos_df, radius=500):
#     m = folium.Map(location=[object_lat, object_lon], zoom_start=17)
    
#     folium.Marker(
#         [object_lat, object_lon],
#         popup=object_name,
#         icon=folium.Icon(color='red', icon='info-sign')
#     ).add_to(m)
#     heat_data = [[row['photo_lat'], row['photo_lon']] for _, row in photos_df.iterrows()]
#     HeatMap(heat_data, radius=15, blur=25).add_to(m)
#     folium.Circle(
#         location=[object_lat, object_lon],
#         radius=radius,
#         color='blue',
#         fill=True,
#         fill_opacity=0.2
#     ).add_to(m)
    
#     return m
# def detect_photo_clusters(photos_df, eps=0.0002, min_samples=3):
#     coords = photos_df[['photo_lat', 'photo_lon']].values
#     db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
#     photos_df['cluster'] = db.labels_
#     return photos_df

# kazan_center = [55.796127, 49.106414]
# m_all = folium.Map(location=kazan_center, zoom_start=13)

# marker_cluster = MarkerCluster().add_to(m_all)

# if len(grouped) == 0:
#     raise ValueError("Нет данных для отображения после группировки")


# for (name, lat, lon), group in grouped:
    
#     if len(group) == 0:
#         print(f"Нет фотографий для объекта: {name}")
#         continue
#     required_columns = ['photo_lat_y', 'photo_lon_y']
#     if not all(col in group.columns for col in required_columns):
#         missing = [col for col in required_columns if col not in group.columns]
#         print(f"Отсутствуют столбцы {missing} для объекта {name}")
#         continue
#     if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
#         print(f"Некорректные координаты объекта {name}: lat={lat}, lon={lon}")
#         continue
#     folium.Marker(
#         [lat, lon],
#         popup=f"<b>{name}</b><br>Фотографий: {len(group)}",
#         icon=folium.Icon(color='red', icon='info-sign')
#     ).add_to(marker_cluster)
#     valid_photos = group[
#         (group['photo_lat_y'].between(-90, 90)) & 
#         (group['photo_lon_y'].between(-180, 180))
#     ]
    
#     if len(valid_photos) == 0:
#         print(f"Нет валидных фотографий для объекта {name}")
#         continue
#     heat_data = valid_photos[['photo_lat_y', 'photo_lon_y']].values.tolist()
#     HeatMap(heat_data, radius=10, blur=20).add_to(marker_cluster)
# m_all


import pandas as pd
import folium
from folium.plugins import HeatMap, Fullscreen, LocateControl, MarkerCluster
import datetime
import h3
from geojson import Feature, Point, FeatureCollection
import json
import matplotlib
import scipy.stats as scs
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import DBSCAN
import numpy as np
import streamlit as st

# Настройка страницы Streamlit
st.set_page_config(layout="wide")

class SeasonalHeatmapVisualizer:
    def __init__(self, data_path="metaData.csv"):
        # Загрузка и подготовка данных
        self.data = pd.read_csv(data_path, parse_dates=['photo_date'])
        self._clean_data()
        self._add_seasons()
        
        # Настройки классов
        self.class_config = {
            1: {"name": "Памятники и артобъекты", "color": "red"},
            2: {"name": "Архитектура", "color": "blue"},
            3: {"name": "Торговые центры и магазины", "color": "purple"},
            4: {"name": "Гастрономия", "color": "pink"},
            5: {"name": "Реки, озера и фонтаны", "color": "lightblue"},
            6: {"name": "Парки и зоны отдыха", "color": "darkgreen"}
        }
        
    def _clean_data(self):
        """Очистка и валидация данных"""
        self.data = self.data.dropna(subset=['photo_lat', 'photo_lon', 'predicted_class'])
        
        # Фильтрация нереальных координат (пример для Казани)
        self.data = self.data[
            (self.data['photo_lat'].between(55.5, 56.0)) & 
            (self.data['photo_lon'].between(48.5, 49.5))
        ]

    def _add_seasons(self):
        """Добавляем информацию о сезонах"""
        self.seasons = {
            'Все сезоны': None,
            'Зима': [12, 1, 2],
            'Весна': [3, 4, 5],
            'Лето': [6, 7, 8],
            'Осень': [9, 10, 11]
        }
        
        self.data['season'] = self.data['photo_date'].apply(
            lambda x: next(
                (season for season, months in self.seasons.items() 
                 if months and x.month in months), 
                None
            )
        )

    def _filter_data(self, season, selected_classes):
        """Фильтрация данных по выбранным параметрам"""
        if season != 'Все сезоны':
            filtered = self.data[self.data['season'] == season]
        else:
            filtered = self.data.copy()
        
        return filtered[filtered['predicted_class'].isin(selected_classes)]

    def _create_map(self, filtered_data):
        """Создание карты с отфильтрованными данными"""
        if filtered_data.empty:
            return None
            
        m = folium.Map(
            location=[filtered_data['photo_lat'].mean(), filtered_data['photo_lon'].mean()],
            zoom_start=14,
            tiles='cartodbpositron',
            control_scale=True
        )
        Fullscreen().add_to(m)
        LocateControl(auto_start=False).add_to(m)
        
        for class_id, config in self.class_config.items():
            class_data = filtered_data[filtered_data['predicted_class'] == class_id]
            if not class_data.empty:
                HeatMap(
                    class_data[['photo_lat', 'photo_lon']].values,
                    radius=30,
                    blur=15,
                    name=config['name'],
                    show=False,
                    min_opacity=0.4,
                    max_zoom=14
                ).add_to(m)
        
        folium.LayerControl(
            position='topright',
            collapsed=False,
            autoZIndex=True
        ).add_to(m)
        
        return m

    def show(self):
        """Отображение визуализации в Streamlit"""
        st.title("Сезонная тепловая карта фотографий")
        
        # Создаем элементы управления
        season = st.selectbox(
            "Сезон:",
            options=list(self.seasons.keys()),
            index=0
        )
        
        selected_classes = []
        cols = st.columns(3)
        for i, (class_id, config) in enumerate(self.class_config.items()):
            with cols[i % 3]:
                if st.checkbox(config['name'], value=True, key=f"class_{class_id}"):
                    selected_classes.append(class_id)
        
        # Фильтрация данных
        filtered_data = self._filter_data(season, selected_classes)
        
        # Отображение информации
        total = len(filtered_data)
        date_range = f"{filtered_data['photo_date'].min().date()} - {filtered_data['photo_date'].max().date()}"
        st.info(f"Всего точек: {total} | Период: {date_range}")
        
        # Создание и отображение карты
        m = self._create_map(filtered_data)
        if m:
            st_folium = st.empty()
            with st_folium:
                folium_static(m, width=1200, height=600)
        else:
            st.warning("Нет данных для отображения с текущими параметрами фильтрации")

def hexagons_dataframe_to_geojson(df_hex, column_name="value"):
    """Конвертация DataFrame с hex-ячейками в GeoJSON"""
    list_features = []
    for i, row in df_hex.iterrows():
        try:
            geometry_for_row = {"type": "Polygon", "coordinates": [h3.cell_to_boundary(h=row["hex_id"])]}
            feature = Feature(geometry=geometry_for_row, id=row["hex_id"], properties={column_name: row[column_name]})
            list_features.append(feature)
        except Exception as e:
            print(f"An exception occurred for hex {row['hex_id']}: {e}")
    feat_collection = FeatureCollection(list_features)
    geojson_result = json.dumps(feat_collection)
    return geojson_result

def get_color(custom_cm, val, vmin, vmax):
    """Получение цвета для карты"""
    return matplotlib.colors.to_hex(custom_cm((val - vmin) / (vmax - vmin)))

def choropleth_map(df_aggreg, column_name="value", border_color='black', fill_opacity=0.7, 
                  color_map_name="cool", initial_map=None, layer_name="Layer"):
    """Создание хороплетной карты"""
    if df_aggreg.empty:
        return initial_map
        
    min_value = df_aggreg[column_name].min()
    max_value = df_aggreg[column_name].max()
    mean_value = df_aggreg[column_name].mean()
    
    if initial_map is None:
        initial_map = folium.Map(location=[55.79, 49.12], zoom_start=11)

    geojson_data = hexagons_dataframe_to_geojson(df_hex=df_aggreg, column_name=column_name)
    custom_cm = matplotlib.cm.get_cmap(color_map_name)

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': get_color(custom_cm, feature['properties'][column_name], vmin=min_value, vmax=max_value),
            'color': border_color,
            'weight': 1,
            'fillOpacity': fill_opacity
        },
        name=layer_name
    ).add_to(initial_map)

    return initial_map

def show_h3_visualization(df):
    """Отображение H3 визуализации в Streamlit"""
    st.title("H3 Визуализация плотности фотографий")
    
    resolution = st.slider("Разрешение H3:", min_value=6, max_value=15, value=9, step=1)
    
    hex_ids = df.apply(lambda row: h3.latlng_to_cell(row['lat'], row['long'], resolution), axis=1)
    df_temp = df.assign(hex_id=hex_ids.values)
    dfbyhexid_temp = df_temp.groupby("hex_id").size().reset_index(name='value')
    
    m = choropleth_map(df_aggreg=dfbyhexid_temp, color_map_name="hot", column_name="value")
    if m:
        folium_static(m, width=1200, height=600)
    else:
        st.warning("Нет данных для отображения")

def show_seasonal_h3_visualization(df):
    """Отображение сезонной H3 визуализации"""
    st.title("Сезонная H3 Визуализация")
    
    resolution = st.slider("Разрешение H3:", min_value=6, max_value=15, value=9, step=1, key="seasonal_resolution")
    
    m = folium.Map(location=[55.79, 49.12], zoom_start=12)
    seasons = ['Зима', 'Весна', 'Лето', 'Осень']
    
    for season in seasons:
        df_season = df[df['season'] == season]
        if not df_season.empty:
            hex_ids = df_season.apply(lambda row: h3.latlng_to_cell(row['lat'], row['long'], resolution), axis=1)
            df_with_resolution = df_season.assign(hex_id=hex_ids.values)
            dfbyhexid = df_with_resolution.groupby("hex_id").size().reset_index(name='value')
            dfbyhexid["percentile"] = dfbyhexid["value"]
            choropleth_map(
                df_aggreg=dfbyhexid, 
                color_map_name="hot", 
                column_name="value", 
                initial_map=m, 
                layer_name=f"{season}"
            )
    
    folium.LayerControl().add_to(m)
    folium_static(m, width=1200, height=600)

def show_object_heatmaps(kazan_meta, predicted_meta):
    """Отображение тепловых карт для объектов"""
    st.title("Тепловые карты объектов")
    
    merged_data = pd.merge(kazan_meta, predicted_meta, on='photo_id', how='inner')
    merged_data = merged_data.head(6000)
    grouped = merged_data.groupby(['osm_object_name', 'osm_lat', 'osm_lon'])
    
    kazan_center = [55.796127, 49.106414]
    m_all = folium.Map(location=kazan_center, zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m_all)
    
    for (name, lat, lon), group in grouped:
        if len(group) == 0:
            continue
            
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            continue
            
        folium.Marker(
            [lat, lon],
            popup=f"<b>{name}</b><br>Фотографий: {len(group)}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(marker_cluster)
        
        valid_photos = group[
            (group['photo_lat_y'].between(-90, 90)) & 
            (group['photo_lon_y'].between(-180, 180))
        ]
        
        if len(valid_photos) > 0:
            heat_data = valid_photos[['photo_lat_y', 'photo_lon_y']].values.tolist()
            HeatMap(heat_data, radius=20, blur=20).add_to(marker_cluster)
    
    folium_static(m_all, width=1200, height=600)

# Основная функция Streamlit
def main():
    # Загрузка данных
    df = pd.read_csv("metaData.csv")
    df = df.rename(columns={"photo_id": "id", "photo_lat": "lat", "photo_lon": "long", 'photo_date': 'date'})
    df = df[['id', 'date', 'lat', 'long', 'predicted_class']].copy()
    df = df.rename(columns={'lat': 'long', 'long': 'lat'})
    df = df.dropna(subset=['lat', 'long', 'predicted_class'])
    
    # Добавление сезонов
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['season'] = df['date'].apply(lambda x: 'Зима' if x.month in [12, 1, 2] else 
                                  'Весна' if x.month in [3, 4, 5] else 
                                  'Лето' if x.month in [6, 7, 8] else 'Осень')
    
    # Создание вкладок
    tab1, tab2, tab3, tab4 = st.tabs([
        "Сезонная тепловая карта", 
        "H3 Визуализация", 
        "Сезонная H3", 
        "Тепловые карты объектов"
    ])
    
    with tab1:
        visualizer = SeasonalHeatmapVisualizer("metaData.csv")
        visualizer.show()
    
    with tab2:
        show_h3_visualization(df)
    
    with tab3:
        show_seasonal_h3_visualization(df)
    
    with tab4:
        kazan_meta = pd.read_csv('kazan_photos_metadata.csv')
        predicted_meta = pd.read_csv('metaData.csv')
        show_object_heatmaps(kazan_meta, predicted_meta)

# Запуск приложения
if __name__ == "__main__":
    from streamlit_folium import folium_static
    main()