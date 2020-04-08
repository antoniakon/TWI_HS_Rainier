name := "RainierHorseshoe"

version := "0.1"

scalaVersion := "2.12.11"

resolvers += Resolver.bintrayRepo("cibotech", "public")

libraryDependencies ++= Seq("com.stripe" %% "rainier-core" % "0.2.2",
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2",
  "com.cibo" %% "evilplot" % "0.6.3",
  "com.stripe" %% "rainier-plot" % "0.1.1"
)