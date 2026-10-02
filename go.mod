module github.com/bornholm/indecis-ner

go 1.27

require (
	github.com/bornholm/go-anon v0.0.0-00010101000000-000000000000
	github.com/bornholm/indecis v0.3.0
	golang.org/x/text v0.42.0
)

replace (
	github.com/bornholm/go-anon => ../go-anon
	github.com/bornholm/indecis => ../indecis
)
