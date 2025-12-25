# Troubleshooting

## Import Errors

**Issue**: Import errors when running examples

**Solutions:**
- Ensure you're running from the project root directory
- Verify that `animations/__init__.py` exists
- Check that `animations/app.cfg` exists

## Configuration Issues

**Issue**: Configuration not loading as expected

**Solutions:**
- Verify all keys in `override()` are UPPERCASE
- Check that `app.cfg` uses correct INI syntax with `=` assignment
- Use `config.to_dict()` to debug current configuration values

## Rendering Errors

**Issue**: Manim rendering errors

**Solutions:**
- Ensure Manim is properly installed: `pip install manim`
- Check that ffmpeg is installed and in PATH
- Try updating Manim: `pip install --upgrade manim`
