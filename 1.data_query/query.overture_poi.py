SELECT *
FROM
    OPENROWSET(
        BULK 'https://overturemapswestus2.blob.core.windows.net/release/2026-04-15.0/theme=places/type=place/',
        FORMAT = 'PARQUET'
    )
WITH
    (
        id VARCHAR(MAX),
        names VARCHAR(MAX),
        categories VARCHAR(MAX),
        confidence VARCHAR(MAX),
        addresses VARCHAR(MAX),
        operating_status VARCHAR(MAX),
        [version] VARCHAR(MAX),
        sources VARCHAR(MAX),
        bbox VARCHAR(200),
        geometry VARBINARY(MAX)
    )
    AS
        [result]
WHERE
        TRY_CONVERT(FLOAT, JSON_VALUE(bbox, '$.xmin')) > -96.621987
    AND TRY_CONVERT(FLOAT, JSON_VALUE(bbox, '$.xmax')) < -94.35339
    AND TRY_CONVERT(FLOAT, JSON_VALUE(bbox, '$.ymin')) > 28.764842
    AND TRY_CONVERT(FLOAT, JSON_VALUE(bbox, '$.ymax')) < 30.90672
