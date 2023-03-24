import os, sys

sys.path.insert(0, "srcPython")
import util
import csv
import datetime
import numpy as np

WEATHER_FILE = util.DATA_DIR + "Weather_Forecast_Tesero.csv"


class Weather:
    def __init__(self, date_time, filename=WEATHER_FILE):
        self.location = None
        self.lat = None
        self.long = None
        self.altitude = None
        self.date_time = None  # Date and time of forecast.
        self.temp = None  # Temp in deg C
        self.dew_point = None  # Dewpoint in deg C
        self.humidity = None  # Humidity as %
        self.wind_gust = None  # Wind Gusts max in m/s (originally Km/h)
        self.wind_speed = None  # Wind Speed in m/s (originally Km/h)
        self.wind_direction = None  # Degrees from north
        self.sea_level_pressure = None  # hPa
        self.solar_radiation = None  # W/m^2 of solar energy (snow melt??)
        self.density = None
        self.filename = filename

        self.__load_data(date_time)
        self.__get_rho()

    def __load_data(self, date_time):

        deltas = []
        dlist = []
        with open(self.filename, "r") as wfile:
            reader = csv.reader(wfile)
            header = next(reader)
            header[0] = util.strip_utf8(header[0])
            header[-1] = util.strip_endline(header[-1])

            for r, row in enumerate(reader):
                date_time_string = row[header.index("datetime")]
                date_time_obj = datetime.datetime.strptime(
                    date_time_string, "%Y-%m-%d %H:%M"
                )
                row[header.index("datetime")] = date_time_obj
                deltas.append(date_time - date_time_obj)
                dlist.append(row)

        dlist = [*dlist[deltas.index(min(deltas))]]
        self.location = dlist[header.index("name")]
        self.date_time = dlist[header.index("datetime")]
        self.temp = np.float64(dlist[header.index("temp (°C)")])
        self.dew_point = np.float64(dlist[header.index("dew point (°C)")])
        self.humidity = np.float64(dlist[header.index("humidity (%)")])
        self.wind_gust = (
            np.float64(dlist[header.index("windgust (max km/h)")]) * 0.277778
        )  # convert to m/s
        self.wind_speed = (
            np.float64(dlist[header.index("windspeed  (Km/h)")]) * 0.277778
        )  # convert to m/s
        self.wind_direction = np.float64(dlist[header.index("wind direction (°)")])
        self.sea_level_pressure = (
            np.float64(dlist[header.index("sealevelpressure (hPa)")]) * 100
        )
        self.solar_radiation = np.float64(dlist[header.index("solarradiation (W/m2)")])

        if self.location == "tesero":
            self.altitude = 913.577
            self.long = 11.51
            self.lat = 46.29

    def __str__(self):
        return """Weather" {t} {d}\n
        Location: \t {self.lat}degN, {self.long}degE {self.altitude} m above sealevel\n
        Temp: \t {self.temp}degC\n
        Dew Point: \t {self.dew_point}degC\n
        Humidity: \t {self.humidity}\n
        Wind: \t {self.wind_speed}m/s at {self.wind_direction} deg with gusts up to {self.wind_gust} \n
        Pressure: \t {self.sea_level_pressure} Pa\n
        Density: \t {self.density} Kg/m^3
        """.format(
            self=self, d=self.date_time.date(), t=self.date_time.time()
        )

    def __get_rho(self):
        # saturation vapour pressure of water...
        if self.temp > 0:
            # Monteith and Unsworth (2008)
            Ps = 0.61078 * np.exp((17.27 * self.temp) / (self.temp + 237.3))
        else:
            # Murray (1967)
            Ps = 0.61078 * np.exp((21.875 * self.temp) / (self.temp + 265.5))

        # Density
        # E. Jones, “The air density equation and the transfer of the mass unit”, J. Res. Natl. Bur. Stand. 83, 1978, pp. 419-428.
        self.density = (0.0034848 / (self.temp + 273.15)) * (
            self.sea_level_pressure - 0.0037960 * self.humidity
        )


if __name__ == "__main__":
    from subject import Subject

    sub = Subject(3)
    print(sub.weather)
