"""Test utility functions."""

from otterapi.codegen.utils import is_url, sanitize_identifier


class TestIsUrl:
    """Test is_url function."""

    def test_valid_http_url(self):
        """Test valid HTTP URLs."""
        assert is_url('http://example.com') is True
        assert is_url('http://example.com/path') is True
        assert is_url('http://example.com:8080') is True
        assert is_url('http://subdomain.example.com') is True

    def test_valid_https_url(self):
        """Test valid HTTPS URLs."""
        assert is_url('https://example.com') is True
        assert is_url('https://api.example.com/v1/openapi.json') is True
        assert is_url('https://example.com:443/path?query=value') is True

    def test_valid_ftp_url(self):
        """Test valid FTP URLs."""
        assert is_url('ftp://ftp.example.com') is True
        assert is_url('ftp://user:pass@ftp.example.com/file.txt') is True

    def test_invalid_urls(self):
        """Test invalid URLs."""
        assert is_url('example.com') is False  # missing scheme
        assert is_url('http://') is False  # missing netloc
        assert is_url('://example.com') is False  # missing scheme
        assert is_url('') is False  # empty string
        assert is_url('not-a-url') is False
        assert is_url('/path/to/file') is False  # local path
        assert is_url('./relative/path') is False  # relative path

    def test_edge_cases(self):
        """Test edge cases for URL validation."""
        assert is_url('file:///path/to/file') is True  # file URL with netloc
        assert is_url('file://') is False  # file URL without path
        assert is_url('http:example.com') is False  # malformed URL
        assert is_url('https://') is False  # missing netloc

    def test_with_special_characters(self):
        """Test URLs with special characters."""
        assert is_url('https://example.com/path%20with%20spaces') is True
        assert is_url('https://example.com/path?query=value&other=data') is True
        assert is_url('https://example.com:8080/path#fragment') is True

    def test_malformed_input(self):
        """Test with malformed input that might cause exceptions."""
        assert is_url(None) is False
        # These should not raise exceptions
        assert is_url('http://[invalid') is False
        assert is_url('http://invalid]') is False


class TestSanitizeIdentifier:
    """Test sanitize_identifier function."""

    def test_simple_names(self):
        """Test sanitizing simple names."""
        assert sanitize_identifier('User') == 'User'
        assert sanitize_identifier('user') == 'user'
        assert sanitize_identifier('MyClass') == 'MyClass'

    def test_names_with_spaces(self):
        """Test sanitizing names with spaces."""
        assert sanitize_identifier('User Profile') == 'UserProfile'
        assert sanitize_identifier('my class name') == 'MyClassName'
        assert sanitize_identifier('  spaced  name  ') == 'SpacedName'

    def test_names_with_hyphens(self):
        """Test sanitizing names with hyphens."""
        assert sanitize_identifier('user-profile') == 'UserProfile'
        assert sanitize_identifier('api-response') == 'ApiResponse'
        assert sanitize_identifier('multi-word-name') == 'MultiWordName'

    def test_names_with_mixed_separators(self):
        """Test sanitizing names with mixed separators."""
        assert sanitize_identifier('user-profile name') == 'UserProfileName'
        assert sanitize_identifier('api response-data') == 'ApiResponseData'
        assert sanitize_identifier('mixed_case-Name Here') == 'MixedCaseNameHere'

    def test_names_with_invalid_characters(self):
        """Test sanitizing names with invalid characters."""
        assert sanitize_identifier('User@Profile') == 'UserProfile'
        assert sanitize_identifier('api#response') == 'ApiResponse'
        assert sanitize_identifier('name$with%symbols') == 'NameWithSymbols'
        assert sanitize_identifier('user.profile') == 'UserProfile'

    def test_names_starting_with_digits(self):
        """Test sanitizing names that start with digits."""
        assert sanitize_identifier('123User') == '_123User'
        assert sanitize_identifier('1st-item') == '_1stItem'
        assert sanitize_identifier('2nd_place') == '_2ndPlace'

    def test_empty_and_none_inputs(self):
        """Test sanitizing empty or None inputs."""
        assert sanitize_identifier('') == 'UnnamedType'
        assert sanitize_identifier('   ') == 'UnnamedType'
        assert sanitize_identifier('___') == 'UnnamedType'

    def test_single_character_names(self):
        """Test sanitizing single character names."""
        assert sanitize_identifier('a') == 'a'
        assert sanitize_identifier('A') == 'A'
        assert sanitize_identifier('1') == '_1'
        assert sanitize_identifier('_') == 'UnnamedType'
        assert sanitize_identifier('@') == 'UnnamedType'

    def test_names_with_underscores(self):
        """Test sanitizing names with underscores."""
        assert sanitize_identifier('user_profile') == 'UserProfile'
        assert sanitize_identifier('_private_var') == 'PrivateVar'
        assert sanitize_identifier('__dunder__') == 'Dunder'

    def test_complex_names(self):
        """Test sanitizing complex names with multiple issues."""
        assert sanitize_identifier('123-user@profile.name') == '_123UserProfileName'
        assert sanitize_identifier('  api-v1.2_response  ') == 'ApiV12Response'
        assert sanitize_identifier('!!!invalid@@@name###') == 'InvalidName'

    def test_pascal_case_conversion(self):
        """Test that names are properly converted to PascalCase."""
        assert sanitize_identifier('snake_case_name') == 'SnakeCaseName'
        assert sanitize_identifier('kebab-case-name') == 'KebabCaseName'
        assert sanitize_identifier('mixed_case-Name') == 'MixedCaseName'

    def test_already_valid_identifiers(self):
        """Test that already valid identifiers are preserved or correctly formatted."""
        assert sanitize_identifier('ValidClassName') == 'ValidClassName'
        assert sanitize_identifier('validVariableName') == 'validVariableName'
        assert sanitize_identifier('CONSTANT_NAME') == 'CONSTANTNAME'

    def test_names_with_numbers(self):
        """Test sanitizing names with numbers in the middle or end."""
        assert sanitize_identifier('user2profile') == 'user2profile'
        assert sanitize_identifier('api-v1-response') == 'ApiV1Response'
        assert sanitize_identifier('item123-name') == 'Item123Name'

    def test_unicode_characters(self):
        """Test sanitizing names with unicode characters."""
        # Unicode characters should be removed
        assert sanitize_identifier('user∆profile') == 'UserProfile'
        assert sanitize_identifier('naïve-user') == 'NaiveUser'
        assert sanitize_identifier('café-name') == 'CafeName'

    def test_very_long_names(self):
        """Test sanitizing very long names."""
        long_name = 'very_long_name_with_many_parts_that_goes_on_and_on'
        expected = 'VeryLongNameWithManyPartsThatGoesOnAndOn'
        assert sanitize_identifier(long_name) == expected

    def test_names_with_only_invalid_characters(self):
        """Test names that consist only of invalid characters."""
        assert sanitize_identifier('@#$%^&*()') == 'UnnamedType'
        assert sanitize_identifier('123456') == '_123456'
        assert sanitize_identifier('!@#$%') == 'UnnamedType'


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_url_and_identifier_together(self):
        """Test using both utilities together in realistic scenarios."""
        # Test case: extracting identifier from URL
        url = 'https://api.example.com/v1/user-profiles'
        if is_url(url):
            path_parts = url.split('/')
            identifier = sanitize_identifier(path_parts[-1])  # "user-profiles"
            assert identifier == 'UserProfiles'

    def test_realistic_openapi_scenarios(self):
        """Test utilities with realistic OpenAPI schema scenarios."""
        # Common OpenAPI schema names that need sanitization
        schema_names = [
            'Pet-Store',
            'user profile',
            'api_response',
            '123-item',
            'User@Info',
            'ResponseData.v1',
        ]

        expected = [
            'PetStore',
            'UserProfile',
            'ApiResponse',
            '_123Item',
            'UserInfo',
            'ResponseDataV1',
        ]

        for name, exp in zip(schema_names, expected):
            assert sanitize_identifier(name) == exp

    def test_url_validation_for_openapi_sources(self):
        """Test URL validation for typical OpenAPI document sources."""
        openapi_urls = [
            'https://petstore.swagger.io/v2/swagger.json',
            'https://api.example.com/openapi.yaml',
            'http://localhost:8080/api/docs/openapi.json',
            'https://raw.githubusercontent.com/user/repo/main/openapi.yaml',
        ]

        for url in openapi_urls:
            assert is_url(url) is True

        openapi_paths = [
            './openapi.yaml',
            '/path/to/openapi.json',
            'openapi.yaml',
            '../schemas/api.json',
        ]

        for path in openapi_paths:
            assert is_url(path) is False
