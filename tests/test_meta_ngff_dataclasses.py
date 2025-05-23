"""Test for NGFF metadata dataclasses."""

from tiatoolbox.wsicore.metadata import ngff


class TestDataclassInit:
    """Test that initialization paths do not error."""

    @staticmethod
    def test_coordinate_transform_defaults() -> None:
        """Test :class:`ngff.CoordinateTransform` init with default args."""
        ngff.CoordinateTransform()

    @staticmethod
    def test_dataset_defaults() -> None:
        """Test :class:`ngff.Dataset` init with default args."""
        ngff.Dataset()

    @staticmethod
    def test_dataset() -> None:
        """Test :class:`ngff.Dataset` init."""
        ngff.Dataset(
            "1",
            coordinateTransformations=[ngff.CoordinateTransform("scale", scale=0.5)],
        )

    @staticmethod
    def test_multiscales_defaults() -> None:
        """Test :class:`ngff.Multiscales` init with default args."""
        ngff.Multiscales()

    @staticmethod
    def test_multiscales_iter() -> None:
        """Test :class:`ngff.Multiscales` init."""
        multiscales = ngff.Multiscales()
        iter_values = list(iter(multiscales))

        # Check if all attributes are present in the yielded values
        assert multiscales.axes in iter_values
        assert multiscales.datasets in iter_values
        assert multiscales.version in iter_values

        # Check the order of yielded values matches __dict__ order
        expected = list(multiscales.__dict__.values())
        assert iter_values == expected

    @staticmethod
    def test_omero_defaults() -> None:
        """Test :class:`ngff.Omero` init with default args."""
        ngff.Omero()

    @staticmethod
    def test_zattrs_defaults() -> None:
        """Test :class:`ngff.Zattrs` init with default args."""
        ngff.Zattrs()
