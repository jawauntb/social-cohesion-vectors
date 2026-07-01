import { PrismaClient, Platform } from "@prisma/client";

const prisma = new PrismaClient();

// Minimal seed: ourselves + a control creator (Lectures on Tap) so the
// pipeline has a held-out competitor to validate generalization against.
async function main() {
  const self = await prisma.competitor.upsert({
    where: { name: "Us (Philosophy)" },
    update: {},
    create: { name: "Us (Philosophy)", platform: Platform.youtube, isSelf: true },
  });

  const control = await prisma.competitor.upsert({
    where: { name: "Lectures on Tap" },
    update: {},
    create: {
      name: "Lectures on Tap",
      platform: Platform.youtube,
      isSelf: false,
      notes: "Control creator: sample weak / okay / strong videos to verify prediction generalizes.",
    },
  });

  console.log("Seeded competitors:", self.name, "+", control.name);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
