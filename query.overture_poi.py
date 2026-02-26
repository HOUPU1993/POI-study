SELECT *
FROM
    OPENROWSET(
        BULK 'https://overturemapswestus2.blob.core.windows.net/release/2025-12-17.0/theme=places/type=place/',
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
        TRY_CONVERT(FLOAT, JSON_VALUE(bbox, '$.xmin')) > -75.195498
    AND TRY_CONVERT(FLOAT, JSON_VALUE(bbox, '$.xmax')) < -71.777492
    AND TRY_CONVERT(FLOAT, JSON_VALUE(bbox, '$.ymin')) > 39.475206
    AND TRY_CONVERT(FLOAT, JSON_VALUE(bbox, '$.ymax')) < 41.527202