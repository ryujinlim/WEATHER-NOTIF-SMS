"""Send a weather summary text message using OpenWeatherMap, OpenAI, and Twilio.

This script fetches the current weather for the configured location, asks OpenAI to
summarize it in friendly prose, and sends that summary as an SMS message via Twilio.

Environment variables:
- OPENWEATHERMAP_API_KEY: API key for OpenWeatherMap.
- WEATHER_CITY: Optional city name (e.g., "Berlin,de").
- WEATHER_LAT and WEATHER_LON: Optional latitude/longitude if WEATHER_CITY is unset.
- WEATHER_UNITS: Optional units for the API (defaults to "metric").
- OPENAI_API_KEY: API key for OpenAI.
- OPENAI_MODEL: Optional model name (defaults to "gpt-4o-mini").
- TWILIO_ACCOUNT_SID: Twilio Account SID.
- TWILIO_AUTH_TOKEN: Twilio Auth Token.
- TWILIO_FROM_NUMBER: Twilio phone number to send from.
- TWILIO_TO_NUMBER: Recipient phone number.

Example usage:
    python weather_notif.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

import requests
from openai import OpenAI
from twilio.rest import Client


WEATHER_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather"


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass
class WeatherConfig:
    api_key: str
    city: str | None
    latitude: str | None
    longitude: str | None
    units: str

    @classmethod
    def from_env(cls) -> "WeatherConfig":
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENWEATHERMAP_API_KEY is required")

        city = os.getenv("WEATHER_CITY")
        lat = os.getenv("WEATHER_LAT")
        lon = os.getenv("WEATHER_LON")

        if not city and not (lat and lon):
            raise ConfigurationError(
                "Provide WEATHER_CITY or both WEATHER_LAT and WEATHER_LON"
            )

        units = os.getenv("WEATHER_UNITS", "metric")
        return cls(api_key=api_key, city=city, latitude=lat, longitude=lon, units=units)


@dataclass
class OpenAIConfig:
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return cls(api_key=api_key, model=model)


@dataclass
class TwilioConfig:
    account_sid: str
    auth_token: str
    from_number: str
    to_number: str

    @classmethod
    def from_env(cls) -> "TwilioConfig":
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")
        to_number = os.getenv("TWILIO_TO_NUMBER")

        missing = [
            name
            for name, value in (
                ("TWILIO_ACCOUNT_SID", account_sid),
                ("TWILIO_AUTH_TOKEN", auth_token),
                ("TWILIO_FROM_NUMBER", from_number),
                ("TWILIO_TO_NUMBER", to_number),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "Missing required Twilio configuration: " + ", ".join(missing)
            )

        return cls(
            account_sid=account_sid,
            auth_token=auth_token,
            from_number=from_number,
            to_number=to_number,
        )


def fetch_weather(config: WeatherConfig) -> Dict[str, Any]:
    """Fetch current weather data from OpenWeatherMap."""
    params: Dict[str, Any] = {
        "appid": config.api_key,
        "units": config.units,
    }

    if config.city:
        params["q"] = config.city
    else:
        params["lat"] = config.latitude
        params["lon"] = config.longitude

    response = requests.get(WEATHER_ENDPOINT, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def build_weather_prompt(data: Dict[str, Any], units: str) -> str:
    """Create a prompt summarizing the weather data for the language model."""
    location = data.get("name") or "the configured location"
    weather = data.get("weather", [{}])[0]
    main = data.get("main", {})
    wind = data.get("wind", {})

    description = weather.get("description", "no description available")
    temperature = main.get("temp")
    feels_like = main.get("feels_like")
    humidity = main.get("humidity")
    pressure = main.get("pressure")
    wind_speed = wind.get("speed")

    parts = [
        "Write a concise, friendly weather update for an SMS message.",
        f"Location: {location}.",
        f"Conditions: {description}.",
    ]

    if temperature is not None:
        parts.append(f"Temperature: {temperature}° ({units}).")
    if feels_like is not None:
        parts.append(f"Feels like: {feels_like}° ({units}).")
    if humidity is not None:
        parts.append(f"Humidity: {humidity}%.")
    if pressure is not None:
        parts.append(f"Pressure: {pressure} hPa.")
    if wind_speed is not None:
        parts.append(f"Wind speed: {wind_speed} m/s.")

    parts.append("Keep the tone warm and clear, and include a simple suggestion if helpful.")
    return "\n".join(parts)


def summarize_weather(openai_cfg: OpenAIConfig, prompt: str) -> str:
    """Ask OpenAI to turn weather data into a human-friendly summary."""
    client = OpenAI(api_key=openai_cfg.api_key)
    response = client.responses.create(
        model=openai_cfg.model,
        input=[
            {
                "role": "system",
                "content": "You write brief, friendly SMS weather updates.",
            },
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=200,
    )

    content = response.output_text.strip()
    if not content:
        raise RuntimeError("OpenAI response was empty")
    return content


def send_sms(twilio_cfg: TwilioConfig, message: str) -> None:
    """Send an SMS message using Twilio."""
    client = Client(twilio_cfg.account_sid, twilio_cfg.auth_token)
    client.messages.create(
        body=message,
        from_=twilio_cfg.from_number,
        to=twilio_cfg.to_number,
    )


def main() -> int:
    try:
        weather_cfg = WeatherConfig.from_env()
        openai_cfg = OpenAIConfig.from_env()
        twilio_cfg = TwilioConfig.from_env()

        weather_data = fetch_weather(weather_cfg)
        prompt = build_weather_prompt(weather_data, weather_cfg.units)
        summary = summarize_weather(openai_cfg, prompt)
        send_sms(twilio_cfg, summary)
        print("Weather summary sent successfully:")
        print(summary)
        return 0
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except requests.HTTPError as exc:
        print(f"Failed to fetch weather data: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - last resort
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
