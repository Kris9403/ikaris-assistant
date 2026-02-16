import psutil

def get_system_stats() -> str:
    """Gets the current CPU usage and battery status of the laptop."""
    cpu_usage = psutil.cpu_percent(interval=1)
    battery = psutil.sensors_battery()
    
    # Use :.1f to round to one decimal place
    percent = f"{battery.percent:.1f}%" if battery else "N/A"
    plugged = "🔌 Plugged in" if battery and battery.power_plugged else "🔋 On Battery"
    
    return f"CPU: {cpu_usage}% | Battery: {percent} ({plugged})"
