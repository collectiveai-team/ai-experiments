def test_package_imports():
    import maintenance_events

    assert maintenance_events.__version__ == "0.1.0"
